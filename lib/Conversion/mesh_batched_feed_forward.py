from source_tree import source_path, logical_name
"""Resident batch-major FFN over two independent SDK collective planes.

The structural IR is shared with tiled FFN. Precision and placement are explicit
source policies. Range safety does not imply numerical accuracy or performance.
"""

import math
from pathlib import Path
import numpy as np
from frontend import check
from feed_forward_ir import canonical
from rms_ir import verify as verify_rms
from input_contracts import (
    effective_bound,
    half_product_bound,
    half_dot_bound,
    validate_batch,
    verify_declarations,
)
from rms_bounds import correlated_output_bound, inverse_bound
from binary16 import quantize, matmul

PROFILE = "mesh_batched_feed_forward.v1"
RMS = dict(
    partition="features",
    axis="y",
    layout="batch_major",
    reduce="sdk_axis",
    result="replicated_columns",
    accumulation="f16",
    collective="f32",
    math="sdk_half",
    compute="dsr",
    fp="relaxed",
)
UP = dict(
    broadcast="resident_rows",
    axis="y",
    reduce="sdk_axis",
    result="feature_columns",
    replicas="rows",
    fusion="collective",
    accumulation="f16",
    collective="f32",
    compute="dsr",
    fp="relaxed",
)
DOWN = dict(
    UP,
    broadcast="resident_columns",
    axis="x",
    result="feature_rows",
    replicas="columns",
    fusion="none",
)
POINT = dict(layout="batch_major", axis="x", compute="dsr", fp="relaxed")


def verify(module, epochs, bound):
    verify_declarations(module)
    m = canonical(module)
    ns = m["nodes"]
    x, gamma, wu, wg, wd, norm, up, gate, act, hidden, down, add, out = ns
    check(
        not m["states"] and not any("place" in n for n in ns),
        "batch FFN owns one resident region",
    )
    check(all(n.get("dtype") == "f16" for n in ns[:-1]), "batch FFN half tensors")
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 2,
        "batch FFN execution bounds",
    )
    verify_rms(
        dict(m, nodes=[x, gamma, norm, dict(out, inputs=[norm["id"]])]), epochs, bound
    )
    check(len(x["shape"]) == len(wu["shape"]) == 2, "batch FFN ranks")
    b, n = x["shape"]
    f = wu["shape"][1]
    check(
        wu["shape"] == wg["shape"] == [n, f] and wd["shape"] == [f, n],
        "batch FFN weight shapes",
    )
    check(
        all(v["shape"] == [b, f] for v in (up, gate, act, hidden))
        and down["shape"] == add["shape"] == [b, n],
        "batch FFN intermediate shapes",
    )
    out.update(dtype="f16", shape=[b, n])
    for v in ns:
        v["interval"] = None
    m.update(profile=PROFILE, epochs=epochs, input_bound=bound)
    plan(m)
    return m


def sum_bound(value, p):
    # Positive bound is identical at every PE. Power-of-two P multiplication is
    # exact in f32, then one f16 rounding. All signed partial sums are bounded
    # by the monotone positive recurrence, independent of SDK reduction order.
    check(
        p in (4, 8) and math.isfinite(value) and 0 <= value * p <= 65504,
        "SDK half I/O collective may overflow",
    )
    return quantize(value * p)


def plan(m, partitions=1):
    x, gamma, wu, wg, wd, norm, up, gate, act, hidden, down, add, out = m["nodes"]
    b, n = x["shape"]
    f = wu["shape"][1]
    p = norm.get("dataflow", {}).get("rows")
    check(partitions == 1 and type(p) is int and p in (4, 8), "batch FFN square region")
    for node, policy in (
        (norm, RMS),
        (up, UP),
        (gate, UP),
        (down, DOWN),
        (act, dict(POINT, compute="map", math="sdk_stable_half")),
        (hidden, POINT),
        (add, dict(POINT, axis="y")),
    ):
        check(
            node.get("dataflow") == dict(policy, rows=p, cols=p),
            "batch FFN explicit precision/placement policy: " + node["op"],
        )
    check(
        all(type(v) is int for v in (b, n, f))
        and 1 <= b <= 16
        and 4 <= n <= 2048
        and 4 <= f <= 4096
        and n % p == f % p == 0
        and n // p % 2 == f // p % 2 == 0
        and n // p <= 512
        and f // p <= 512,
        "batch FFN even bounded feature shards",
    )
    nt, ft = n // p, f // p
    mode = m.get("instrumentation", "sampled")
    check(mode in ("sampled", "counters"), "batch FFN instrumentation")
    padded = (b + 1) // 2 * 2
    packed = 2 * b * ft
    capacity = max(padded, packed, b * nt)
    check(
        max(nt * ft, packed, b * nt, capacity) <= 32767, "batch FFN descriptor extent"
    )
    vb, gb = [effective_bound(v, m["input_bound"]) for v in (x, gamma)]
    square = half_product_bound(vb, vb)
    local = half_dot_bound(square, 1.0, nt)
    total = sum_bound(local, p)
    inv = inverse_bound(total, n, norm["epsilon"])
    norm_bound = correlated_output_bound(vb, gb, total, n, norm["epsilon"])
    from rms_l1_bounds import normalized_l1, projection as projection_bound

    l1 = normalized_l1(vb, gb, total, n, nt, p, norm["epsilon"])
    check(l1["elementwise_bound"] == norm_bound, "same finite normalized envelope")
    projection = [
        projection_bound(l1["bound"], effective_bound(w, m["input_bound"]), nt, p)
        for w in (wu, wg)
    ]
    # Exhaustive fixed SDK stable-SiLU probe establishes |silu(x)| <= |x|
    # for all finite half encodings, including exp underflow and signed zero.
    product = half_product_bound(projection[0]["reduced"], projection[1]["reduced"])
    dl = half_dot_bound(product, effective_bound(wd, m["input_bound"]), ft)
    delta = sum_bound(dl, p)
    check(delta + vb <= 65504, "batch FFN residual range")
    final = quantize(delta + vb)
    # All live numeric and observation buffers are explicit. No aliasing is
    # assumed; runtime/linked SDK code and stack are conservatively reserved.
    alloc = dict(
        X=2 * b * nt,
        gamma=2 * nt,
        weights=6 * nt * ft,
        normalized=2 * b * nt,
        square_scratch=2 * b * nt,
        sums=2 * padded,
        projections=2 * packed,
        activation=2 * b * ft,
        hidden=2 * b * ft,
        delta=2 * b * nt,
        result=2 * b * nt,
        collective_send=4 * capacity,
        collective_reduced=4 * capacity,
        rms_history=4 * padded if mode == "sampled" else 2,
        projection_partial=2 * packed if mode == "sampled" else 2,
        down_partial=2 * b * nt if mode == "sampled" else 2,
    )
    memory = dict(alloc, protocol_descriptors=1024, code_stack_reserve=16384)
    check(sum(memory.values()) <= 49152, "batch FFN full PE memory budget")
    resources = dict(
        colors=[0, 1, 4, 5],
        input_queues=[2, 3, 4, 5],
        output_queues=[2, 3, 4, 5],
        local_tasks=[10, 14, 15, 16, 17],
        microthreads=[],
        dsr_ownership="SDK X bank-set1 and Y bank-set2; synchronous local DSR1/2 only after provider callback; next provider starts after local return",
        completion="SDK broadcast callback releases widened send/reduced storage before caller continuation; no route-axis reassignment or global barrier inferred",
    )
    from sdk2101_resources import check_default_memcpy

    resources["sdk_default_memcpy"] = check_default_memcpy(resources)
    schedule = dict(
        profile=PROFILE,
        P=p,
        rows=p,
        cols=p,
        B=b,
        N=n,
        F=f,
        Nt=nt,
        Ft=ft,
        padded_batches=padded,
        packed_length=packed,
        collective_capacity=capacity,
        epochs=m["epochs"],
        epsilon=norm["epsilon"],
        instrumentation=mode,
        memory_per_pe=memory,
        numeric_allocations=alloc,
        resources=resources,
        numerical_bounds=dict(
            square=square,
            local_sum=local,
            reduced_sum=total,
            inverse=inv,
            normalized=norm_bound,
            normalized_l1=l1,
            projections=projection,
            activation=projection[1]["reduced"],
            hidden=product,
            down_local=dl,
            delta=delta,
            result=final,
            scope="Finite-range proof only; per-stage original-input accuracy gates required",
        ),
        input_bindings={
            v["host"]: role
            for v, role in zip(m["nodes"][:5], ("X", "gamma", "wu", "wg", "wd"))
        },
        output_port=out["host"],
        stages=[
            "local half RMS sum",
            "SDK Y f32 reduce/broadcast and half narrow",
            "half RMS normalize and two local DSR projections",
            "fused SDK Y f32 reduce/broadcast",
            "stable half SiLU and half product; local DSR DOWN",
            "SDK X f32 reduce/broadcast",
            "half residual; host command-stream completion",
        ],
        ownership=dict(
            input="feature Y, replicated X",
            weights="UP/GATE[Y-input,X-hidden]; DOWN[X-hidden,Y-output]",
            hidden="feature X, replicated Y",
            output="feature Y, replicated X",
        ),
        nodes=[
            dict(id=f"p{xx}_{yy}", tile=[xx, yy], place=[4 + xx, 1 + yy])
            for yy in range(p)
            for xx in range(p)
        ],
    )

    from batched_ffn_lifetimes import storage_plan

    schedule["storage_lifetimes"] = storage_plan(schedule)
    return schedule


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "batch FFN epochs")
    q = lambda x: np.asarray(x, np.float16).astype(float)
    results = []
    for batch in batches:
        validate_batch(m, batch)
        x, gamma, wu, wg, wd = [
            np.asarray(batch[v["host"]], float).reshape(v["shape"])
            for v in m["nodes"][:5]
        ]
        norm = q(
            x
            * gamma
            / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + m["nodes"][5]["epsilon"])
        )
        up, gate = matmul(norm, wu), matmul(norm, wg)
        exp = np.exp(-np.abs(gate))
        activation = q(np.where(gate >= 0, gate / (1 + exp), gate * exp / (1 + exp)))
        result = q(x + matmul(q(up * activation), wd))
        results.append({m["nodes"][-1]["host"]: result.ravel().tolist()})
    return results, {}


def generate(s, dest):
    runtime = source_path("runtime")
    for name, source in {
        "layout.csl": "batched_ffn_layout.csl",
        "pe.csl": "batched_ffn_pe.csl",
        "batched_rms_local.csl": "batched_rms_local.csl",
        "batched_matmul_local.csl": "batched_matmul_local.csl",
        "sdk_axis_reduce.csl": "sdk_axis_reduce.csl",
        "sdk_stable_silu.csl": "sdk_stable_silu.csl",
    }.items():
        (Path(dest) / name).write_bytes((runtime / source).read_bytes())
