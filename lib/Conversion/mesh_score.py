from source_tree import source_path, logical_name
"""Transpose-view contraction with resident Q, vertical K exchange and rotating roots."""

import copy
from pathlib import Path
from inference_resources import compute, score_exchange, score_reduce
import numpy as np
from frontend import check
from half_matrix import inputs
from binary16 import matmul
from mesh_common import pack_tiles, unpack_tiles
from mesh_twohop import cycle

POLICY = dict(
    exchange="vertical_two_hop",
    reduce="rotating_root",
    order="east_first",
    overlap="double_buffer",
    compute="dsr",
    fp="relaxed",
)


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    nodes = m["nodes"]
    by = {n["id"]: n for n in nodes}
    check(
        len(nodes) == len(by) == 5
        and sorted(n["op"] for n in nodes)
        == ["input", "input", "matmul", "output", "transpose"],
        "score typed five-node contraction",
    )
    check(all(v in by for n in nodes for v in n["inputs"]), "score defined operands")
    t = next(n for n in nodes if n["op"] == "transpose")
    op = next(n for n in nodes if n["op"] == "matmul")
    out = next(n for n in nodes if n["op"] == "output")
    check(
        len(t["inputs"]) == 1 and len(op["inputs"]) == 2 and op["inputs"][1] == t["id"],
        "score right operand transpose view",
    )
    q = by[op["inputs"][0]]
    k = by[t["inputs"][0]]
    check(
        q["op"] == k["op"] == "input"
        and q["id"] != k["id"]
        and q["host"] != k["host"]
        and not q["inputs"]
        and not k["inputs"],
        "score distinct input tensors",
    )
    check(
        out["inputs"] == [op["id"]]
        and not m["states"]
        and not any("place" in n for n in nodes),
        "score result edge and region ownership",
    )
    check(
        all(n.get("dtype") == "f16" for n in (q, k, t, op)),
        "score binary16 view/operands/result",
    )
    shape = q["shape"]
    check(
        len(shape) == 2 and all(type(v) is int and 1 <= v <= 256 for v in shape),
        "score matrix bounds",
    )
    M, N = shape
    check(
        k["shape"] == shape and t["shape"] == [N, M] and op["shape"] == [M, M],
        "score logical transpose and contraction extents",
    )
    check(
        "dataflow" in op and all("dataflow" not in n for n in (q, k, t, out)),
        "score policy on contraction, transpose is a view",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 1,
        "score epoch/input bounds",
    )
    out.update(shape=[M, M], dtype="f16")
    m["nodes"] = [q, k, t, op, out]
    for n in nodes:
        n["interval"] = None
    m.update(profile="mesh_score.v1", epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    q, k, t, op, out = m["nodes"]
    d = op["dataflow"]
    p = d.get("rows")
    M, N = q["shape"]
    check(
        set(d) == set(POLICY) | {"rows", "cols"}
        and all(d[v] == x for v, x in POLICY.items()),
        "score explicit vertical/root policy",
    )
    order = cycle(p)
    check(
        partitions == 1 and d["cols"] == p and M % p == N % p == 0,
        "score divisible square region",
    )
    mt, nt = M // p, N // p
    L, S = mt * nt, mt * mt
    check(
        L % 4 == S % 4 == 0 and max(p * L, p * S) <= 32767,
        "score four-half packing and signed offsets",
    )
    mode = m.get("instrumentation", "sampled")
    check(mode in ("sampled", "counters"), "score instrumentation")
    memory = dict(
        resident_buffers=2 * (4 * L + 2 * S),
        observations=2 * p * (L + S) if mode == "sampled" else 4,
        control=2 * p + 128,
        sdk_code_stack_reserve=16384,
    )
    check(sum(memory.values()) <= 49152, "score PE memory budget")
    return dict(
        profile="mesh_score.v1",
        M=M,
        N=N,
        P=p,
        rows=p,
        cols=p,
        Mt=mt,
        Nt=nt,
        length=L,
        score_length=S,
        epochs=m["epochs"],
        instrumentation=mode,
        memory_per_pe=memory,
        cycle=order,
        transpose=dict(
            kind="logical_view",
            materialized=False,
            input_layout="column-major token/feature tiles",
        ),
        resources=dict(
            colors=list(range(1, 12)),
            input_queues=[3, 4, 5, 6, 7],
            output_queues=[3, 4, 5, 6, 7],
            active_input_queues=[3, 4, 5],
            active_output_queues=[3, 4, 5],
            microthreads=[2, 3],
            local_tasks=[19, 20, 25, 26],
            compute_dsrs=[1],
            reduction_dsrs=[1, 2],
            async_communication_dsrs=[4, 6],
            explicit_dsr_phases=dict(
                local_compute=compute(),
                overlapped_exchange=score_exchange(),
                joined_reduction=score_reduce(),
            ),
            reuse="local FMA completes before synchronous root reduction reuses DSR1",
        ),
        stages=[
            "copy immutable K to private send storage",
            "rotate K vertically while local QK-transpose partial computes",
            "reduce partial east-first to current K token root",
            "join vertical send/receive and compute/reduction completion before swap",
        ],
        root_rule="cycle[(position(y)-round)%P]",
        nodes=[
            dict(id=f"p{x}_{y}", tile=[x, y], place=[4 + x, 1 + y])
            for y in range(p)
            for x in range(p)
        ],
    )


def reference(s, q, k):
    p, mt, nt = s["P"], s["Mt"], s["Nt"]
    order = cycle(p)
    qt = pack_tiles(q, p, p, "F")
    kt = pack_tiles(k, p, p, "F")
    partial = np.zeros((p, p, p, mt * mt))
    owners = np.zeros((p, p, p, mt * nt))
    roots = np.zeros((p, p, p), np.uint16)
    out = np.zeros((p, p, mt * mt))
    half = lambda v: np.asarray(v, np.float16).astype(float)
    for y in range(p):
        for step in range(p):
            root = order[(order.index(y) - step) % p]
            roots[y, :, step] = root
            for x in range(p):
                owners[y, x, step] = kt[root, x]
                partial[y, x, step] = matmul(
                    qt[y, x].reshape(mt, nt, order="F"),
                    kt[root, x].reshape(mt, nt, order="F").T,
                ).ravel(order="F")
            v = partial[y, :, step]
            value = v[root].copy()
            if root < p - 1:
                east = v[-1].copy()
                for x in range(p - 2, root, -1):
                    east = half(east + v[x])
                value = half(value + east)
            if root > 0:
                west = v[0].copy()
                for x in range(1, root):
                    west = half(west + v[x])
                value = half(value + west)
            out[y, root] = value
    return (
        partial.reshape(p, p, -1),
        owners.reshape(p, p, -1),
        roots,
        unpack_tiles(out, mt, mt, "F"),
    )


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "score epoch count")
    return [
        {m["nodes"][-1]["host"]: matmul(q, k.T).ravel().tolist()}
        for q, k in [inputs(m, b) for b in batches]
    ], {}


def accuracy(q, k, actual):
    nominal = q @ k.T
    err = actual - nominal
    norm = float(np.linalg.norm(nominal))
    peak = float(np.max(np.abs(nominal)))
    l2 = float(np.linalg.norm(err)) / max(norm, 1e-30)
    scaled = float(np.max(np.abs(err))) / max(peak, 1e-30)
    check(
        np.all(np.isfinite(actual)) and l2 <= 0.015 and scaled <= 0.02,
        "score standard normwise accuracy",
    )
    return dict(
        contract="score-half-normwise-v1",
        fixed_accuracy_passed=True,
        relative_l2=l2,
        peak_scaled_error=scaled,
    )


def generate(s, dest):
    rt = source_path("runtime")
    dest = Path(dest)
    for name in ("layout", "pe"):
        (dest / (name + ".csl")).write_bytes(
            (rt / ("score_" + name + ".csl")).read_bytes()
        )
    for name in ("inference_comm.csl", "inference_routes.csl"):
        (dest / name).write_bytes((rt / name).read_bytes())
    (dest / "WaferLLM-LICENSE.txt").write_bytes(
        (rt / "waferllm-LICENSE.txt").read_bytes()
    )
    (dest / "SOURCE-NOTICE.txt").write_text(
        "Source-derived score_matmul/matmul_T_compute and unchanged communication library from MeshInfra/WaferLLM fd1c2daae37cd68706c03fc8009887ecee9900f8, Apache-2.0. Typed transpose view, immutable K with private working copy, optional partial/live-owner observations and lifecycle counters. No host transpose, full attention or hardware claim.\n"
    )
