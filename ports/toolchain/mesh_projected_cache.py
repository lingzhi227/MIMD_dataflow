"""Verified normalized QKV/pair/read-only-cache composition over shared SDK planes."""

import copy, math
from pathlib import Path
import numpy as np
from frontend import check
from projected_cache_ir import canonical
from input_contracts import (
    verify_declarations,
    effective_bound,
    validate_batch,
    half_product_bound,
    half_dot_bound,
)
from rms_l1_bounds import normalized_l1
from blocked_projection_bounds import bound as projection
from pair_rotation_bounds import bound as pair_bound
from binary16 import quantize
from mesh_batched_feed_forward import RMS, UP
from mesh_cache_attention import verify as cache_verify, plan as cache_plan

PROFILE = "mesh_projected_cache.v1"
PAIR = dict(
    partition="features",
    axis="x",
    layout="batch_major",
    coefficients="feature_pairs",
    compute="dsr",
    fp="relaxed",
)


def cache_module(m):
    ns = m["nodes"]
    x, gamma, wq, wk, wv, c, s, key, value, wo = ns[:10]
    used = {n.get("host") for n in ns}
    host = "__computed_query"
    while host in used:
        host += "_"
    query = dict(
        id=ns[14]["id"],
        op="input",
        inputs=[],
        host=host,
        dtype="f16",
        shape=ns[14]["shape"][:],
        abs_bound=0,
    )
    return dict(
        copy.deepcopy(m), nodes=copy.deepcopy([x, query, key, value, wo, *ns[16:23]])
    )


def verify(module, epochs, bound):
    verify_declarations(module)
    m = canonical(module)
    ns = m["nodes"]
    check(
        not m["states"] and not any("place" in n for n in ns),
        "projected cache single region",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 2,
        "projected cache execution domain",
    )
    check(
        all(n.get("dtype") == "f16" for n in ns[:22]),
        "projected cache half tensor graph",
    )
    check(
        all(
            isinstance(n.get("shape"), list)
            and len(n["shape"]) == 2
            and all(type(v) is int and v > 0 for v in n["shape"])
            for n in ns[:22]
        ),
        "projected cache positive rank-two shapes",
    )
    b, n = ns[0]["shape"]
    seq = ns[7]["shape"][0]
    check(
        ns[1]["shape"] == [1, n]
        and all(v["shape"] == [n, n] for v in ns[2:5])
        and ns[5]["shape"] == ns[6]["shape"] == [1, n // 2],
        "projected cache normalized projection/coefficient shapes",
    )
    check(
        all(v["shape"] == [b, n] for v in ns[10:16]),
        "projected cache prefix intermediate shapes",
    )
    for i, node in enumerate(ns[22:]):
        node.update(shape=[b, n], dtype="f16")
    for node in ns:
        node["interval"] = None
    m.update(profile=PROFILE, epochs=epochs, input_bound=bound)
    cache_verify(cache_module(m), epochs, bound)
    plan(m)
    return m


def plan(m, partitions=1):
    ns = m["nodes"]
    x, gamma, wq, wk, wv, c, s, key, value, wo = ns[:10]
    norm, q, k, v, rq, rk = ns[10:16]
    b, n = x["shape"]
    p = norm.get("dataflow", {}).get("rows")
    check(
        partitions == 1 and type(p) is int and p in (4, 8, 16),
        "projected cache square SDK region",
    )
    check(
        norm.get("dataflow") == dict(RMS, rows=p, cols=p),
        "projected cache SDK normalized dataflow",
    )
    check(
        type(norm.get("epsilon")) in (int, float)
        and math.isfinite(norm["epsilon"])
        and 0 < float(np.float16(norm["epsilon"])) <= 1,
        "projected cache positive half epsilon",
    )
    check(n % p == 0 and n // p % 2 == 0, "projected cache paired local features")
    nt = n // p
    for node in (q, k, v):
        check(
            node.get("dataflow") == dict(UP, rows=p, cols=p)
            and node.get("accumulation") == "block_f32"
            and type(node.get("block_size")) is int
            and 1 <= node["block_size"] <= nt
            and nt % node["block_size"] == 0,
            "projected cache fused projections use explicit divisible local half blocks",
        )
    for node in (rq, rk):
        check(
            node.get("dataflow") == dict(PAIR, rows=p, cols=p)
            and node.get("pair_order") == "odd_even",
            "projected cache explicit source pairs",
        )
    vb, gb = [effective_bound(v, m["input_bound"]) for v in (x, gamma)]
    check(max(vb, gb) <= 1, "projected cache original activation/gamma domain")
    total = quantize(p * half_dot_bound(half_product_bound(vb, vb), 1, nt))
    l1 = normalized_l1(vb, gb, total, n, nt, p, norm["epsilon"])
    projections = [
        projection(
            l1["bound"], effective_bound(w, m["input_bound"]), nt, p, node["block_size"]
        )
        for w, node in zip((wq, wk, wv), (q, k, v))
    ]
    cb, sb = [effective_bound(v, m["input_bound"]) for v in (c, s)]
    check(max(cb, sb) <= 1, "projected cache supplied coefficient bounds")
    pairs = [pair_bound(v["reduced"], cb, sb) for v in projections[:2]]
    tail = cache_verify(cache_module(m), m["epochs"], m["input_bound"])
    tail["instrumentation"] = m.get("instrumentation", "sampled")
    result = cache_plan(
        tail,
        partitions,
        derived_query_bound=pairs[0]["output_absolute"],
        use_probability_mass=True,
    )
    check(result["P"] == p, "projected cache consistent prefix/tail mesh")
    pad = result["padded_batches"]
    mode = result["instrumentation"]
    cap = max(3 * b * nt, b * result["St"], pad)
    alloc = dict(result["numeric_allocations"])
    alloc.update(
        gamma=2 * nt,
        qkv_weights=6 * nt * nt,
        cosine=nt,
        sine=nt,
        normalized=2 * b * nt,
        rms_scratch=2 * b * nt,
        rms_sums=2 * pad,
        rms_history=4 * pad if mode == "sampled" else 2,
        projections=6 * b * nt,
        projection_partial=6 * b * nt if mode == "sampled" else 2,
        rotated_key=2 * b * nt,
        pair_scratch=4 * nt,
        query_pair_history=4 * b * nt if mode == "sampled" else 2,
        key_pair_history=4 * b * nt if mode == "sampled" else 2,
        collective_send=4 * cap,
        collective_reduced=4 * cap,
    )
    for role, node in zip(("query", "key", "new_value"), (q, k, v)):
        alloc["qkv_" + role + "_block_partial"] = 2 * (
            nt if node["block_size"] < nt else 1
        )
        alloc["qkv_" + role + "_block_total"] = 4 * (
            nt if node["block_size"] < nt else 1
        )
    memory = dict(alloc, protocol_descriptors=1024, code_stack_reserve=20480)
    check(
        max(3 * b * nt, 2 * b * nt, 3 * nt * nt) <= 32767
        and sum(memory.values()) <= 49152,
        "projected cache full graph descriptor/PE memory budget",
    )
    result.update(
        profile=PROFILE,
        epsilon=norm["epsilon"],
        projection_blocks=[node["block_size"] for node in (q, k, v)],
        collective_capacity=cap,
        numeric_allocations=alloc,
        memory_per_pe=memory,
        auxiliary_ports=dict(new_key=ns[23]["host"], new_value=ns[24]["host"]),
        input_bindings={
            a["host"]: role
            for a, role in zip(
                ns[:10],
                ("X", "gamma", "wq", "wk", "wv", "cosine", "sine", "K", "V", "W"),
            )
        },
    )
    result["numerical_bounds"].update(
        rms_sum=total, rms_l1=l1, projections=projections, pairs=pairs
    )
    result["ownership"].update(
        query="Computed normalized QKV branch, feature X/replica Y; pair result feeds score directly",
        new_key="Computed paired K branch output, feature X/replica Y; not appended",
        new_value="Computed V branch output, feature X/replica Y; not consumed by old cache attention",
        residual="Original X feature Y remains immutable",
    )
    result["stages"] = [
        "local RMS squares/sums",
        "SDK Y RMS SUM",
        "normalize/local QKV",
        "SDK Y fused QKV SUM",
        "Q/K pairs/local score",
        *result["stages"][1:],
    ]
    result["resources"][
        "dsr_ownership"
    ] = "SDK X bank1/Y bank2 callbacks release local ownership before synchronous RMS/matmul or pair banks1..5; no overlapping operations"
    result["resources"]["extent_transitions"] = [
        pad,
        3 * b * nt,
        b * result["St"],
        pad,
        pad,
        b * nt,
        b * nt,
    ]
    # Detailed callback-scoped leases are validated by the composition's storage layer.
    from projected_cache_lifetimes import storage_plan

    result["storage_lifetimes"] = storage_plan(result)
    return result


def values(m, batch):
    validate_batch(m, batch)
    check(
        set(batch) == {n["host"] for n in m["nodes"][:10]},
        "projected cache input names",
    )
    arrays = []
    for n in m["nodes"][:10]:
        a = np.asarray(batch[n["host"]], float).reshape(n["shape"])
        check(
            np.all(np.isfinite(a))
            and np.all(np.abs(a) <= effective_bound(n, m["input_bound"]))
            and np.array_equal(a, a.astype(np.float16).astype(float)),
            "projected cache exact bounded half inputs",
        )
        arrays.append(a)
    return arrays


def evaluate(m, batches):
    from blocked_matmul import evaluate as blocked
    from mesh_pair_rotation import reference as rotate

    check(len(batches) == m["epochs"], "projected cache native epoch count")
    qh = lambda a: np.asarray(a, np.float16).astype(float)
    out = []
    ns = m["nodes"]
    for batch in batches:
        x, gamma, wq, wk, wv, c, s, key, value, wo = values(m, batch)
        norm = qh(
            x
            * gamma
            / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + ns[10]["epsilon"])
        )
        q, k, v = [
            blocked(norm, w, ns[i]["block_size"])
            for i, w in zip((11, 12, 13), (wq, wk, wv))
        ]
        rq, rk = [qh(rotate(a, c, s, "odd_even", False)[1]) for a in (q, k)]
        scores = blocked(rq, key.T, ns[17]["block_size"]) * ns[18]["scale"]
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        prob = qh(e / e.sum(axis=1, keepdims=True))
        context = blocked(prob, value, ns[19]["block_size"])
        delta = blocked(context, wo, ns[20]["block_size"])
        result = qh(x + delta)
        out.append(
            {
                node["host"]: a.ravel().tolist()
                for node, a in zip(ns[22:], (result, rk, v))
            }
        )
    return out, {}


def generate(s, dest):
    rt = Path(__file__).parent / "runtime"
    dest = Path(dest)
    mapping = {
        "layout.csl": "projected_cache_layout.csl",
        "pe.csl": "projected_cache_pe.csl",
    }
    mapping.update(
        {
            name: name
            for name in (
                "batched_rms_local.csl",
                "batched_matmul_local.csl",
                "batched_matmul_blocked.csl",
                "batched_pair_rotation_local.csl",
                "batched_softmax_local.csl",
                "sdk_axis_reduce.csl",
                "sdk_axis_max.csl",
            )
        }
    )
    for name, source in mapping.items():
        (dest / name).write_bytes((rt / source).read_bytes())
