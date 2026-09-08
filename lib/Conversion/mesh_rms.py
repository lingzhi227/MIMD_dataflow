from source_tree import source_path, logical_name
"""Typed rowwise RMS normalization; algorithm and target arithmetic kept separate."""

import copy, math
from pathlib import Path
import numpy as np
from frontend import check
from half_matrix import inputs
from sdk_math_reference import rms_inverse_f16

POLICY = dict(
    partition="tiles",
    reduce="bidirectional_chain",
    weights="feature_columns",
    accumulation="f16",
    math="sdk_half",
    compute="dsr",
    fp="relaxed",
)


def verify(module, epochs, bound):
    from rms_ir import verify as verify_rms

    m = verify_rms(module, epochs, bound)
    m["profile"] = "mesh_rms.v1"
    plan(m)
    return m


def plan(m, partitions=1):
    a, w, op, out = m["nodes"]
    d = op["dataflow"]
    rows = d["rows"]
    cols = d["cols"]
    M, N = a["shape"]
    check(
        set(d) == set(POLICY) | {"rows", "cols"}
        and all(d[k] == v for k, v in POLICY.items()),
        "RMS explicit supported dataflow policy",
    )
    check(
        partitions == 1
        and type(rows) is int
        and type(cols) is int
        and rows in (2, 4, 8, 16)
        and cols in (4, 8, 16),
        "RMS bounded rectangular region",
    )
    check(
        type(M) is int
        and type(N) is int
        and 1 <= M <= 512
        and 4 <= N <= 4096
        and M % rows == N % cols == 0,
        "RMS tile divisibility and dimensions",
    )
    mt, nt = M // rows, N // cols
    mode = m.get("instrumentation", "sampled")
    check(mode in ("sampled", "counters"), "RMS instrumentation")
    memory = dict(
        data=6 * mt * nt + 2 * nt + 2 * mt,
        observations=6 * mt if mode == "sampled" else 2,
        protocol=128,
        code_stack_reserve=16384,
    )
    check(
        mt * nt <= 32767 and sum(memory.values()) <= 49152,
        "RMS DSD and PE memory budget",
    )
    return dict(
        profile=m["profile"],
        rows=rows,
        cols=cols,
        M=M,
        N=N,
        Mt=mt,
        Nt=nt,
        epsilon=op["epsilon"],
        instrumentation=mode,
        epochs=m["epochs"],
        dtype="f16",
        memory_per_pe=memory,
        stages=[
            dict(
                kind="row_local_sum",
                map="square",
                combine="add",
                rounding="f16_each_operator",
            ),
            dict(
                kind="row_allreduce",
                topology="bidirectional_chain",
                root=cols // 2,
                root_order=["east", "west"],
            ),
            dict(kind="row_map", operator="reciprocal_sdk_sqrt", epsilon=op["epsilon"]),
            dict(
                kind="column_weight_then_row_scale", result_layout="column_major_tiles"
            ),
        ],
        resources=dict(
            colors=list(range(4, 9)),
            input_queues=[3, 4, 6],
            output_queues=[3, 4, 6],
            compute_dsr=1,
            reduction_src1_dsr=2,
            local_tasks=[],
            microthreads="synchronous fabric operations; no asynchronous user tasks",
            host_bindings="fixed X/W/result cells",
        ),
        nodes=[
            dict(id=f"p{x}_{y}", tile=[x, y], place=[4 + x, 1 + y])
            for y in range(rows)
            for x in range(cols)
        ],
    )


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "RMS epochs")
    output = []
    for b in batches:
        x, w = inputs(m, b)
        v = (
            x
            * w
            / np.sqrt(
                np.sum(x * x, axis=1)[:, None] / x.shape[1] + m["nodes"][2]["epsilon"]
            )
        )
        check(np.all(np.isfinite(v)), "RMS nominal finite output")
        output.append(
            {m["nodes"][3]["host"]: v.astype(np.float16).astype(float).ravel().tolist()}
        )
    return output, {}


def reference(s, x, w):
    """Target-order oracle: local half squares, source east-first chain, SDK math."""
    q = lambda a: np.asarray(a, dtype=np.float16).astype(float)
    rows, cols, mt, nt = s["rows"], s["cols"], s["Mt"], s["Nt"]
    local = np.zeros((rows, cols, mt))
    for y in range(rows):
        for col in range(cols):
            for j in range(nt):
                local[y, col] = q(
                    local[y, col] + q(x[y * mt : (y + 1) * mt, col * nt + j] ** 2)
                )
    left = local[:, 0].copy()
    for col in range(1, cols // 2):
        left = q(left + local[:, col])
    right = local[:, -1].copy()
    for col in range(cols - 2, cols // 2, -1):
        right = q(right + local[:, col])
    total = q(q(local[:, cols // 2] + right) + left)
    check(
        np.all(np.isfinite(total)),
        "RMS half sum overflow; use wider accumulation policy",
    )
    inv = np.asarray(
        [[rms_inverse_f16(v, s["N"], s["epsilon"]) for v in row] for row in total]
    )
    result = q(q(x * w) * inv.reshape(s["M"], 1))
    check(np.all(np.isfinite(result)), "RMS half result overflow")
    return (
        local,
        np.repeat(total[:, None, :], cols, axis=1),
        np.repeat(inv[:, None, :], cols, axis=1),
        result,
    )


def generate(s, dest):
    for name in ("layout", "pe", "row_reduce"):
        (Path(dest) / (name + ".csl")).write_text(
            (source_path("runtime") / ("rms_" + name + ".csl")).read_text()
        )
