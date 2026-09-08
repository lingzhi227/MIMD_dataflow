from source_tree import source_path, logical_name
"""Shared normalized SSA producer, resident 2D weights, fused grouped reduction.

Placement follows explicit policies; neither source/application nor port names
select arithmetic. The source collective keeps one set of routes across extents.
"""

import copy
from pathlib import Path
import numpy as np
from frontend import check
from normalized_fanout_ir import canonical
import mesh_batched_rms as rms
from decode_grouped_reference import grouped
from input_contracts import validate_batch
from binary16 import matmul
from input_contracts import effective_bound, half_dot_bound

POLICY = dict(
    broadcast="resident_rows",
    reduce="grouped_two_tree",
    result="feature_columns",
    replicas="rows",
    fusion="collective",
    compute="dsr",
    fp="relaxed",
)


def rms_module(m):
    v = copy.deepcopy(m)
    norm = v["nodes"][2]
    v["nodes"] = v["nodes"][:3] + [
        dict(
            id="__rms_observer",
            op="output",
            inputs=[norm["id"]],
            host="__rms_observer",
            dtype="f16",
            shape=norm["shape"][:],
        )
    ]
    return v


def branches(m):
    return [m["nodes"][i : i + 3] for i in range(3, len(m["nodes"]), 3)]


def verify(module, epochs, bound):
    m = canonical(module)
    base = rms.verify(rms_module(m), epochs, bound)
    m["nodes"][:3] = base["nodes"][:3]
    b, n = m["nodes"][0]["shape"]
    for w, op, out in branches(m):
        check(
            w.get("dtype") == op.get("dtype") == "f16"
            and len(w["shape"]) == 2
            and w["shape"][0] == n
            and op["shape"] == [b, w["shape"][1]],
            "batched fanout typed matmul dimensions",
        )
        check(
            not w["inputs"] and not any("place" in v for v in (w, op, out)),
            "batched fanout region owns placement",
        )
        out.update(dtype="f16", shape=op["shape"][:])
    for v in m["nodes"]:
        v["interval"] = None
    m.update(profile="mesh_batched_fanout.v1", epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    s = rms.plan(rms_module(m), partitions)
    bs = branches(m)
    c = len(bs)
    f = bs[0][0]["shape"][1]
    p, b, nt = s["P"], s["B"], s["Nt"]
    check(
        type(f) is int and f > 0 and f % p == 0 and (f // p) % 2 == 0,
        "batched fanout packed even output shard",
    )
    ft = f // p
    fused = c * b * ft
    for w, op, out in bs:
        d = op.get("dataflow", {})
        check(
            set(d) == set(POLICY) | {"rows", "cols", "groups"}
            and all(d[k] == v for k, v in POLICY.items()),
            "batched fanout explicit resident fused policy",
        )
        check(
            d["rows"] == d["cols"] == p
            and d["groups"] == s["groups"]
            and w["shape"] == [s["N"], f],
            "batched fanout matching physical schedules",
        )
    padded = (fused + 1) // 2 * 2
    memory = copy.deepcopy(s["memory_per_pe"])
    memory["projection_weights"] = 2 * c * nt * ft
    memory["projection_results"] = 2 * padded
    memory["projection_witnesses"] = (
        2 * padded if s["instrumentation"] == "sampled" else 2
    )
    memory["dynamic_descriptors"] = 256
    check(
        c * nt * ft <= 32767 and padded <= 32767 and sum(memory.values()) <= 49152,
        "batched fanout DSD/PE memory budget",
    )
    limits = []
    for w, _, _ in bs:
        local = half_dot_bound(
            s["numerical_bounds"]["output"], effective_bound(w, m["input_bound"]), nt
        )
        _, total = grouped(
            np.full((p, 1), local), s["group_size"], s["root_within_group"]
        )
        check(
            np.isfinite(total[0]) and total[0] <= 65504,
            "batched fanout finite grouped output bound",
        )
        limits.append(dict(local=local, reduced=float(total[0])))
    s.update(
        profile=m["profile"],
        F=f,
        Ft=ft,
        projections=c,
        packed_length=fused,
        padded_projection=padded,
        collective_capacity=max(padded, s["padded_batches"]),
        memory_per_pe=memory,
        branch_bindings=[
            dict(
                index=i,
                weight=w["host"],
                output=o["host"],
                node=op["id"],
                offset=i * b * ft,
                length=b * ft,
            )
            for i, (w, op, o) in enumerate(bs)
        ],
    )
    s["numerical_bounds"]["projections"] = limits
    s["ownership"][
        "projection"
    ] = "weights[y-feature,x-output], packed [branch,batch,local-output]; outputs replicated across Y rows"
    s["stages"] = s["stages"][:-1] + [
        "local: all independent projections reuse normalized input and DSR1",
        "collective: one packed branch-major Y reduction using same routes and queues",
        "host: command-stream completion on every PE",
    ]
    from sdk2101_resources import check_default_memcpy

    s["resources"]["sdk_default_memcpy"] = check_default_memcpy(s["resources"])
    from batched_resources import leases, storage_plan

    s["resources"]["explicit_dsr_leases"] = leases(c)
    s["storage_lifetimes"] = storage_plan(s)
    s["resources"]["extent_transitions"] = [s["padded_batches"], padded]
    s["resources"][
        "extent_ordering"
    ] = "all PEs execute identical blocking collective sequence; only lengths and memory bases change, no route reconfiguration or global join inferred"
    return s


def inputs(m, b):
    validate_batch(m, b)
    return [
        np.asarray(b[n["host"]], float).reshape(n["shape"])
        for n in m["nodes"]
        if n["op"] == "input"
    ]


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "batched fanout epochs")
    result = []
    for batch in batches:
        values = inputs(m, batch)
        x, w = values[:2]
        norm = np.asarray(
            x
            * w
            / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + m["nodes"][2]["epsilon"]),
            np.float16,
        ).astype(float)
        result.append(
            {
                out["host"]: matmul(norm, q).ravel().tolist()
                for (_, _, out), q in zip(branches(m), values[2:])
            }
        )
    return result, {}


def reference(s, x, w, weights):
    local, sums, inv, norm = rms.reference(s, x, w)
    partial = np.zeros((s["P"], s["P"], s["padded_projection"]))
    for y in range(s["P"]):
        for col in range(s["P"]):
            for i, q in enumerate(weights):
                tile = matmul(
                    norm[:, y * s["Nt"] : (y + 1) * s["Nt"]],
                    q[
                        y * s["Nt"] : (y + 1) * s["Nt"],
                        col * s["Ft"] : (col + 1) * s["Ft"],
                    ],
                )
                off = i * s["B"] * s["Ft"]
                partial[y, col, off : off + tile.size] = tile.ravel()
    _, total = grouped(partial, s["group_size"], s["root_within_group"])
    outputs = []
    for i in range(s["projections"]):
        off = i * s["B"] * s["Ft"]
        outputs.append(
            total[:, off : off + s["B"] * s["Ft"]]
            .reshape(s["P"], s["B"], s["Ft"])
            .transpose(1, 0, 2)
            .reshape(s["B"], s["F"])
        )
    return local, sums, norm, partial, total, outputs


def generate(s, dest):
    from grouped_collective_csl import generate as collective

    dest = Path(dest)
    rt = source_path("runtime")
    for name, src in [
        ("layout.csl", "batched_fanout_layout.csl"),
        ("pe.csl", "batched_fanout_pe.csl"),
        ("batched_rms_local.csl", "batched_rms_local.csl"),
        ("batched_matmul_local.csl", "batched_matmul_local.csl"),
    ]:
        (dest / name).write_bytes((rt / src).read_bytes())
    collective(dest)
    (dest / "WaferLLM-LICENSE.txt").write_bytes(
        (rt / "waferllm-LICENSE.txt").read_bytes()
    )
    (dest / "SOURCE-NOTICE.txt").write_text(
        "Derived from WaferLLM Decode fd1c2daae37cd68706c03fc8009887ecee9900f8 vecmat_computation and grouped QKV/ZZ fusion, Apache-2.0. Shared corrected RMS, resident 2D weights, dynamic padded extents on one unchanged Y collective. This is a normalized projection subgraph, not full Decode or cache management.\n"
    )
