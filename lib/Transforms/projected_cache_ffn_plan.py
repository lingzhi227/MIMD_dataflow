"""Parent-derived ranges and explicit resident storage for attention plus FFN.

Produces a candidate schedule. Linked target memory, numerical error gates and
source-relative performance remain obligations of generated-code qualification.
"""

from frontend import check
from projected_cache_ffn_ir import canonical
from mesh_projected_cache import verify as verify_attention, plan as plan_attention
from rms_statistic_ir import mean_boundary
from blocked_projection_bounds import bound as projection_bound
from blocked_matmul import finite_bound
from input_contracts import half_product_bound, effective_bound
from binary16 import quantize
from sdk2101_resources import check_default_memcpy


def plan(module, epochs=8, bound=2):
    from input_contracts import verify_declarations

    verify_declarations(module)
    graph = canonical(module)
    nodes = graph["module"]["nodes"]
    check(
        not module["states"] and not any("place" in n for n in nodes),
        "composed resident region owns all placements",
    )
    check(
        all(v.get("dtype") == "f16" for v in nodes if v["op"] != "output"),
        "composed graph half tensor types",
    )
    inputs = [v for v in nodes if v["op"] == "input"]
    check(
        len(inputs) == len({v["host"] for v in inputs}) == 13,
        "composed distinct input hosts",
    )
    check(
        module.get("instrumentation", "sampled") == "sampled",
        "composed candidate observation policy",
    )
    attention = verify_attention(graph["attention"], epochs, bound)
    graph["attention"] = attention
    graph["module"].update(epochs=epochs, input_bound=bound)
    s = plan_attention(attention)
    x, gamma, wu, wg, wd, norm, up, gate, act, hidden, down, add, out = graph["ffn"][
        "nodes"
    ]
    by = {n["id"]: n for n in graph["module"]["nodes"]}
    z = by[graph["boundary"]["producer"]]
    b, n, p = s["B"], s["N"], s["P"]
    nt = s["Nt"]
    f = wu["shape"][1]
    check(type(f) is int and 4 <= f <= 4096, "composed FFN bounded hidden dimension")
    ft = f // p
    check(
        p == 16 and f % p == 0 and ft % 2 == 0 and 1 <= ft <= 512,
        "composed FFN 16-way even shard",
    )
    check(
        wu["shape"] == wg["shape"] == [n, f] and wd["shape"] == [f, n],
        "composed FFN matrix shapes",
    )
    check(
        all(v["shape"] == [b, f] for v in (up, gate, act, hidden))
        and down["shape"] == add["shape"] == [b, n],
        "composed FFN result shapes",
    )
    # This edge is derived here from the actual verified parent, never an input stub.
    boundary = mean_boundary(
        norm, z, gamma, derived_input_bound=s["numerical_bounds"]["result"]
    )
    from mesh_batched_feed_forward import UP, DOWN, POINT

    for node, policy in (
        (up, UP),
        (gate, UP),
        (down, DOWN),
        (act, dict(POINT, compute="map", math="sdk_stable_half")),
        (hidden, POINT),
        (add, dict(POINT, axis="y")),
    ):
        check(
            node.get("dataflow") == dict(policy, rows=p, cols=p),
            "composed FFN explicit operation policy",
        )
    for node, inner in ((up, nt), (gate, nt), (down, ft)):
        check(
            node.get("accumulation") == "block_f32"
            and type(node.get("block_size")) is int
            and 1 <= node["block_size"] <= inner
            and inner % node["block_size"] == 0,
            "composed FFN blocked contractions",
        )
    projections = [
        projection_bound(
            boundary["normalized_range"]["l1_bound"],
            effective_bound(w, bound),
            nt,
            p,
            node["block_size"],
        )
        for w, node in ((wu, up), (wg, gate))
    ]
    product = half_product_bound(projections[0]["reduced"], projections[1]["reduced"])
    local_delta = finite_bound(
        product, effective_bound(wd, bound), ft, down["block_size"]
    )
    check(p * local_delta <= 65504, "composed FFN DOWN finite global sum")
    delta = quantize(p * local_delta)
    final = delta + s["numerical_bounds"]["result"]
    check(final <= 65504, "composed FFN finite residual")
    pad = s["padded_batches"]
    cap = max(s["collective_capacity"], 2 * b * ft, b * nt, pad)
    alloc = dict(s["numeric_allocations"])
    alloc.update(collective_send=4 * cap, collective_reduced=4 * cap)
    alloc.update(
        ffn_weights=6 * nt * ft,
        ffn_normalized=2 * b * nt,
        ffn_square_scratch=2 * b * nt,
        ffn_sums=2 * pad,
        ffn_projections=4 * b * ft,
        ffn_activation=2 * b * ft,
        ffn_hidden=2 * b * ft,
        ffn_delta=2 * b * nt,
        ffn_result=2 * b * nt,
        ffn_rms_history=4 * pad,
        ffn_projection_partial=4 * b * ft,
        ffn_down_partial=2 * b * nt,
        mean_send=4 * pad,
        mean_reduced=4 * pad,
    )
    for role, width, node, inner in (
        ("up", ft, up, nt),
        ("gate", ft, gate, nt),
        ("down", nt, down, ft),
    ):
        length = width if node["block_size"] < inner else 1
        alloc["ffn_" + role + "_block_partial"] = 2 * length
        alloc["ffn_" + role + "_block_total"] = 4 * length
    memory = dict(alloc, protocol_descriptors=1024, code_stack_reserve=28672)
    check(sum(memory.values()) <= 49152, "composed FFN candidate PE budget")
    resources = dict(s["resources"])
    resources["local_tasks"] = sorted(resources["local_tasks"] + [12])
    resources["sdk_default_memcpy"] = check_default_memcpy(resources)
    result = dict(
        profile="projected_cache_ffn.candidate",
        attention=s,
        boundary=boundary,
        F=f,
        Ft=ft,
        collective_capacity=cap,
        numeric_allocations=alloc,
        memory_per_pe=memory,
        resources=resources,
        numerical_bounds=dict(
            projections=projections,
            hidden=product,
            down_local=local_delta,
            delta=delta,
            result=quantize(final),
        ),
        graph=graph,
        status="parent-derived range and storage only; complete backend/ELF/numerical/performance qualification pending",
    )

    result["output_bindings"] = {
        out["host"]: dict(physical="ffn_result", axis="y", width=n, offset=0),
        attention["nodes"][23]["host"]: dict(
            physical="rotated_key", axis="x", width=n, offset=0
        ),
        attention["nodes"][24]["host"]: dict(
            physical="projections", axis="x", width=n, offset=2 * b * nt
        ),
    }
    from projected_cache_ffn_lifetimes import storage_plan

    result["storage_lifetimes"] = storage_plan(result)
    return result
