"""Logical-tile matmul with both alignments on device and strided RHS access."""

import copy
from pathlib import Path
from inference_resources import projection_lease
import numpy as np
from frontend import check
from mesh_common import verify_matmul, pack_tiles, unpack_tiles
from half_matrix import inputs, evaluate
from mesh_twohop import cycle, block_index

POLICY = dict(
    exchange="two_hop",
    initial_align="both_axes",
    reduce="local",
    overlap="double_buffer",
    fp="relaxed",
    compute="dsr",
)


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    by = {n["id"]: n for n in m["nodes"]}
    check(
        len(by) == len(m["nodes"]) == 4
        and sorted(n["op"] for n in m["nodes"])
        == ["input", "input", "matmul", "output"],
        "device matmul unique typed contraction",
    )
    check(
        all(v in by for n in m["nodes"] for v in n["inputs"]),
        "device matmul defined operands",
    )
    op = next(n for n in m["nodes"] if n["op"] == "matmul")
    out = next(n for n in m["nodes"] if n["op"] == "output")
    check(len(op["inputs"]) == 2, "device matmul two operands")
    m["nodes"] = [by[v] for v in op["inputs"]] + [op, out]
    m = verify_matmul(m, epochs, bound, compute_modes=("dsr",))
    a, b, op, out = m["nodes"]
    check(
        all(n.get("dtype") == "f16" for n in (a, b, op)) and bound <= 1,
        "device matmul finite half input bound0..1",
    )
    check(
        a["shape"][0] == a["shape"][1] and max(a["shape"] + b["shape"]) <= 256,
        "device matmul bounded square left operand",
    )
    out["dtype"] = "f16"
    m["profile"] = "mesh_device_matmul.v1"
    plan(m)
    return m


def plan(m, partitions=1):
    a, b, op, out = m["nodes"]
    d = op["dataflow"]
    p = d.get("rows")
    order = cycle(p)
    M, N = a["shape"][0], b["shape"][1]
    check(
        set(d) == set(POLICY) | {"rows", "cols"}
        and all(d[k] == v for k, v in POLICY.items()),
        "device matmul both-axis policy",
    )
    check(
        partitions == 1 and d["cols"] == p and M % p == N % p == 0,
        "device matmul divisible square region",
    )
    mt, nt = M // p, N // p
    S, L = mt * mt, mt * nt
    check(
        S % 4 == L % 4 == 0 and max(p * S, p * L) <= 32767,
        "device matmul packed communication and signed offsets",
    )
    mode = m.get("instrumentation", "sampled")
    check(mode in ("sampled", "counters"), "device matmul observation mode")
    memory = dict(
        resident_buffers=2 * (3 * S + 4 * L),
        observations=2 * p * (S + 2 * L) if mode == "sampled" else 6,
        control=128,
        sdk_code_stack_reserve=16384,
    )
    check(sum(memory.values()) <= 49152, "device matmul PE memory budget")
    return dict(
        profile="mesh_device_matmul.v1",
        M=M,
        N=N,
        P=p,
        rows=p,
        cols=p,
        Mt=mt,
        Nt=nt,
        left_length=S,
        length=L,
        epochs=m["epochs"],
        instrumentation=mode,
        cycle=order,
        memory_per_pe=memory,
        stages=[
            "copy immutable logical input tiles into private working buffers",
            "vertical RHS column-dependent prealignment",
            "horizontal LHS row-dependent prealignment",
            "overlap both-axis exchange with local DSR FMA through strided RHS view",
            "join both directions before next buffer swap",
        ],
        layout=dict(
            host="logical column-major tiles, no host prealignment",
            rhs_access="feature stride Mt, per-contraction offset1; no local transpose buffer",
        ),
        resources=dict(
            colors=list(range(1, 12)),
            input_queues=[3, 4, 5, 6, 7],
            output_queues=[3, 4, 5, 6, 7],
            active_input_queues=[5, 7],
            active_output_queues=[5, 7],
            microthreads=[0, 1, 2, 3],
            local_tasks=[19, 20, 25, 26],
            compute_dsrs=[1],
            communication_dsrs=[3, 4, 5, 6],
            explicit_dsr_lease=projection_lease(),
        ),
        k_block_rule="cycle[(position(y)+position(x)-round)%P]",
        nodes=[
            dict(id=f"p{x}_{y}", tile=[x, y], place=[4 + x, 1 + y])
            for y in range(p)
            for x in range(p)
        ],
    )


def reference(s, a, b):
    p, mt, nt = s["P"], s["Mt"], s["Nt"]
    ap = pack_tiles(a, p, p, "F")
    bp = pack_tiles(b, p, p, "F")
    history = np.zeros((p, p, p, mt * nt))
    left = np.zeros((p, p, p, mt * mt))
    right = np.zeros_like(history)
    out = np.zeros((p, p, mt * nt))
    for y in range(p):
        for x in range(p):
            acc = np.zeros((mt, nt))
            for step in range(p):
                block = block_index(p, y, x, step)
                aa = ap[y, block].reshape(mt, mt, order="F")
                bb = bp[block, x].reshape(mt, nt, order="F")
                left[y, x, step] = ap[y, block]
                right[y, x, step] = bp[block, x]
                for k in range(mt):
                    acc = np.asarray(
                        acc + aa[:, k, None] * bb[None, k, :], np.float16
                    ).astype(float)
                history[y, x, step] = acc.ravel(order="F")
            out[y, x] = acc.ravel(order="F")
    return (
        history.reshape(p, p, -1),
        left.reshape(p, p, -1),
        right.reshape(p, p, -1),
        unpack_tiles(out, mt, nt, "F"),
    )


def accuracy(a, b, actual):
    nominal = a @ b
    err = actual - nominal
    l2 = float(np.linalg.norm(err)) / max(float(np.linalg.norm(nominal)), 1e-30)
    peak = float(np.max(np.abs(err))) / max(float(np.max(np.abs(nominal))), 1e-30)
    check(
        np.all(np.isfinite(actual)) and l2 <= 0.015 and peak <= 0.02,
        "device matmul standard normwise accuracy",
    )
    return dict(
        contract="device-matmul-half-normwise-v1",
        fixed_accuracy_passed=True,
        relative_l2=l2,
        peak_scaled_error=peak,
    )


def generate(s, dest):
    rt = Path(__file__).parent / "runtime"
    dest = Path(dest)
    for name in ("layout", "pe"):
        (dest / (name + ".csl")).write_bytes(
            (rt / ("device_matmul_" + name + ".csl")).read_bytes()
        )
    for name in ("inference_comm.csl", "inference_routes.csl"):
        (dest / name).write_bytes((rt / name).read_bytes())
    (dest / "WaferLLM-LICENSE.txt").write_bytes(
        (rt / "waferllm-LICENSE.txt").read_bytes()
    )
    (dest / "SOURCE-NOTICE.txt").write_text(
        "Source-derived output_matmul from MeshInfra/WaferLLM fd1c2daae37cd68706c03fc8009887ecee9900f8, Apache-2.0. Explicit device vertical RHS prealignment and strided DSD view accept logical column-major input; source arithmetic/communication retained. Immutable inputs use private copies. No host prealignment/local transpose copy or full attention/hardware claim.\n"
    )
