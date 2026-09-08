"""Feature-sharded, batch-major RMS with source grouped reduction and replicas."""

from pathlib import Path
import math
import numpy as np
from frontend import check
from rms_ir import verify as verify_rms
from mesh_rms import evaluate
from half_matrix import inputs
from input_contracts import effective_bound, half_product_bound, half_dot_bound
from rms_bounds import inverse_bound, correlated_output_bound
from decode_grouped_reference import grouped
from sdk_math_reference import rms_inverse_f16

POLICY = dict(
    partition="features",
    axis="y",
    layout="batch_major",
    reduce="grouped_two_tree",
    result="replicated_columns",
    accumulation="f16",
    math="sdk_half",
    compute="dsr",
    fp="relaxed",
)
FILES = {
    "layout.csl": "batched_rms_layout.csl",
    "pe.csl": "batched_rms_pe.csl",
    "batched_rms_local.csl": "batched_rms_local.csl",
    "axis_grouped_reduce.csl": "axis_grouped_reduce.csl",
}


def verify(module, epochs, bound):
    m = verify_rms(module, epochs, bound)
    m["profile"] = "mesh_batched_rms.v1"
    plan(m)
    return m


def plan(m, partitions=1):
    x, w, op, out = m["nodes"]
    d = op["dataflow"]
    check(
        set(d) == set(POLICY) | {"rows", "cols", "groups"}
        and all(d[k] == v for k, v in POLICY.items()),
        "batched RMS explicit feature-axis/layout policy",
    )
    p, g = d["rows"], d["groups"]
    check(
        partitions == 1 and type(p) is int and p in (4, 8) and d["cols"] == p,
        "batched RMS square 4/8 mesh",
    )
    check(
        type(g) is int and g >= 2 and p % g == 0 and p // g >= 2,
        "batched RMS grouped trees",
    )
    b, n = x["shape"]
    check(
        type(b) is int
        and 1 <= b <= 16
        and type(n) is int
        and 4 <= n <= 2048
        and n % p == 0
        and n // p <= 512,
        "batched RMS batches and feature divisibility",
    )
    mode = m.get("instrumentation", "sampled")
    check(mode in ("sampled", "counters"), "batched RMS instrumentation")
    nt = n // p
    padded = (b + 1) // 2 * 2
    size = p // g
    r1 = size // 2
    r2 = (g // 2) * size + r1
    vb, wb = effective_bound(x, m["input_bound"]), effective_bound(w, m["input_bound"])
    square = half_product_bound(vb, vb)
    local = half_dot_bound(square, 1.0, nt)
    groups, total = grouped(np.full((p, 1), local), size, r1)
    total = float(total[0])
    check(math.isfinite(total) and total <= 65504, "batched RMS grouped sum range")
    inv = inverse_bound(total, n, op["epsilon"])
    output = correlated_output_bound(vb, wb, total, n, op["epsilon"])
    memory = dict(
        data=6 * b * nt + 2 * nt + 2 * padded + 2,
        observations=4 * padded if mode == "sampled" else 2,
        protocol=256,
        code_stack_reserve=16384,
    )
    check(
        b * nt <= 32767 and sum(memory.values()) <= 49152, "batched RMS DSD/PE memory"
    )
    return dict(
        profile=m["profile"],
        rows=p,
        cols=p,
        P=p,
        B=b,
        N=n,
        Nt=nt,
        padded_batches=padded,
        length=b * nt,
        groups=g,
        group_size=size,
        root_within_group=r1,
        global_root=r2,
        epsilon=op["epsilon"],
        epochs=m["epochs"],
        instrumentation=mode,
        memory_per_pe=memory,
        numerical_bounds=dict(
            square=square,
            local_sum=local,
            group_sums=groups.ravel().tolist(),
            reduced_sum=total,
            inverse=inv,
            output=output,
            scope="Monotone source grouped half sum, exhaustive pinned SDK inverse model and correlated gamma-first output range; not accuracy proof",
        ),
        ownership=dict(
            features="y contiguous shards",
            batches="all batches resident on every PE; batch-major within a feature shard",
            replicas="identical x columns",
            padding="zero extra half lane if batch count is odd; not a logical batch",
        ),
        stages=[
            "init: configure Y routes on a quiescent mesh",
            "local: half square and stationary memory DSR accumulation",
            "collective: grouped sum then broadcast of padded batch vector",
            "local: original X times gamma then per-batch SDK RMS scale",
            "host: command-stream completion on every PE",
        ],
        resources=dict(
            colors=[5, 6, 7, 8, 9],
            input_queues=[3, 4, 5, 6, 7],
            output_queues=[3, 4, 5, 6, 7],
            local_tasks=[],
            microthreads=[],
            compute_dsrs=[1, 2],
            collective_src1_dsr=2,
            leases="synchronous local phase ends before collective uses src1 DSR2; local normalize resumes after collective return",
            reconfiguration="only init_task before any launch; routes remain Y during all warm calls",
            join="collective return is a local dependency, not mesh-wide quiescence; in-call axis switching is not admitted; blocking SDK call across all PEs precedes next host launch",
        ),
        nodes=[
            dict(
                id=f"p{x}_{y}",
                tile=[x, y],
                place=[4 + x, 1 + y],
                feature_start=y * nt,
                replica=x,
            )
            for y in range(p)
            for x in range(p)
        ],
    )


def reference(s, x, w):
    q = lambda a: np.asarray(a, np.float16).astype(float)
    local = np.zeros((s["P"], s["padded_batches"]))
    for y in range(s["P"]):
        for j in range(s["Nt"]):
            local[y, : s["B"]] = q(local[y, : s["B"]] + q(x[:, y * s["Nt"] + j] ** 2))
    groups, total = grouped(local, s["group_size"], s["root_within_group"])
    inv = np.array([rms_inverse_f16(v, s["N"], s["epsilon"]) for v in total[: s["B"]]])
    result = q(q(x * w) * inv[:, None])
    return local, total, inv, result


def generate(s, dest):
    dest = Path(dest)
    for name, source in FILES.items():
        (dest / name).write_bytes(
            (Path(__file__).parent / "runtime" / source).read_bytes()
        )
