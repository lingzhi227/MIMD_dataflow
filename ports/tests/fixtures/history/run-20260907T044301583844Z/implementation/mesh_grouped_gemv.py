"""Row-vector contraction with two-stage grouped half reduction and replication."""

from pathlib import Path
from frontend import check
from mesh_common import verify_matmul, pack_tiles
from half_matrix import inputs, evaluate


def verify(module, epochs, bound):
    m = verify_matmul(module, epochs, bound, compute_modes=("dsr",))
    a, b, op, out = m["nodes"]
    d = op["dataflow"]
    check(
        all(n.get("dtype") == "f16" for n in (a, b, op)),
        "grouped contraction requires binary16",
    )
    check(
        set(d)
        == {"rows", "cols", "broadcast", "reduce", "groups", "result", "fp", "compute"},
        "grouped contraction policy fields",
    )
    check(
        d["broadcast"] == "host_rows"
        and d["reduce"] == "grouped_two_tree"
        and d["result"] == "replicated_columns",
        "grouped contraction ownership policy",
    )
    check(a["shape"][0] == 1, "grouped row-vector contraction")
    out["dtype"] = "f16"
    m["profile"] = "mesh_grouped_gemv.v1"
    plan(m)
    return m


def plan(m, partitions=1):
    a, b, op, out = m["nodes"]
    d = op["dataflow"]
    p = d["rows"]
    g = d["groups"]
    check(
        partitions == 1
        and type(p) is int
        and type(d["cols"]) is int
        and p in (4, 8)
        and d["cols"] == p,
        "grouped square4/8 ownership",
    )
    check(
        type(g) is int and g >= 2 and p % g == 0 and p // g >= 2,
        "group count and nonempty two-sided phase",
    )
    k, n = b["shape"]
    check(k % p == n % p == 0, "grouped tile divisibility")
    mt, nt = k // p, n // p
    size = p // g
    r1 = size // 2
    r2 = (g // 2) * size + r1
    check(max(mt * nt, 4 * nt) <= 32767, "grouped signed DSD bounds")
    mode = m.get("instrumentation", "sampled")
    check(mode in ("sampled", "counters"), "grouped instrumentation mode")
    memory = dict(
        operands=2 * (mt + mt * nt),
        result_and_observations=2 * nt * 4 if mode == "sampled" else 2 * nt + 2,
        protocol_and_timing=128,
        sdk_code_tasks_stack_reserve=16384,
    )
    check(sum(memory.values()) <= 49152, "grouped PE memory budget")
    return dict(
        profile=m["profile"],
        instrumentation=mode,
        P=p,
        M=k,
        N=n,
        Mt=mt,
        Nt=nt,
        groups=g,
        group_size=size,
        root_within_group=r1,
        global_root=r2,
        epochs=m["epochs"],
        dtype="f16",
        stages=[
            "replicate X segment across columns; row-major W tiles",
            "DSR1 local fused half contraction",
            "within-group half reduction toward midpoint",
            "group-root half reduction toward global midpoint",
            "broadcast result to every row",
        ],
        memory_per_pe=memory,
        resources=dict(
            colors=[5, 6, 7, 8, 9, 12, 13, 14, 15, 16],
            input_queues=list(range(2, 8)),
            output_queues=list(range(2, 8)),
            compute_dsr=1,
            reduction_src1_dsr=2,
            host_bindings="fixed X/W/result cells",
        ),
        nodes=[
            dict(
                id=f"p{x}_{y}",
                tile=[x, y],
                place=[4 + x, 1 + y],
                group=y // size,
                phase1_root=y % size == r1,
                phase2_root=y == r2,
            )
            for y in range(p)
            for x in range(p)
        ],
    )


def pack(a, b, p):
    import numpy as np

    mt = b.shape[0] // p
    return np.repeat(np.asarray(a).reshape(p, 1, mt), p, axis=1), pack_tiles(
        b, p, p, "C"
    )


def reduce_group(values, root):
    """Source tree: both sides chain inward; root consumes lower side before upper."""
    import numpy as np

    def add(a, b):
        return np.asarray(
            np.asarray(a, dtype=float) + np.asarray(b, dtype=float), np.float16
        ).astype(float)

    top = values[0]
    for i in range(1, root):
        top = add(top, values[i])
    result = values[root]
    if root + 1 < len(values):
        bottom = values[-1]
        for i in range(len(values) - 2, root, -1):
            bottom = add(bottom, values[i])
        result = add(bottom, result)
    return add(top, result)


def reference(s, a, b):
    import numpy as np
    from binary16 import matmul

    p, mt, nt = s["P"], s["Mt"], s["Nt"]
    size = s["group_size"]
    local = np.asarray(
        [
            [
                matmul(
                    a[:, y * mt : (y + 1) * mt],
                    b[y * mt : (y + 1) * mt, x * nt : (x + 1) * nt],
                )[0]
                for x in range(p)
            ]
            for y in range(p)
        ]
    )
    groups = np.asarray(
        [
            reduce_group(local[start : start + size], s["root_within_group"])
            for start in range(0, p, size)
        ]
    )
    total = reduce_group(groups, s["groups"] // 2)
    return local, groups, total


def generate(s, dest):
    root = Path(__file__).parent / "runtime"
    dest = Path(dest)
    for name in ["pe", "layout", "comm", "routes"]:
        text = (root / ("grouped_" + name + ".csl")).read_text()
        if name == "pe" and s["instrumentation"] == "counters":
            for statement in [
                "timestamp.get_timestamp(&hls_compute_start);",
                "timestamp.get_timestamp(&hls_compute_end);",
                "const d=@increment_dsd_offset(hls_hist_dsd,phase*Nt,f16);",
                "@fmovh(d,res_dsd);",
                "const hls_hist_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{Nt}->hls_history[i]});",
            ]:
                check(
                    text.count(statement) == 1, "grouped diagnostic site " + statement
                )
                text = text.replace(statement, "")
            text = text.replace(
                "var hls_history=@zeros([3*Nt]f16);", "var hls_history=@zeros([1]f16);"
            )
        filename = (
            "grouped_" + name + ".csl" if name in ("comm", "routes") else name + ".csl"
        )
        (dest / filename).write_text(text)
