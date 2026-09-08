from source_tree import source_path, logical_name
"""Typed half matrix contraction over a two-hop cycle with double-buffer overlap."""

from pathlib import Path
from frontend import check
from mesh_common import verify_matmul, pack_tiles
from half_matrix import inputs, evaluate


def cycle(p):
    check(type(p) is int and p in (4, 8), "two-hop qualified even mesh4/8")
    return list(range(0, p, 2)) + list(range(p - 1, 0, -2))


def block_index(p, y, x, step=0):
    order = cycle(p)
    pos = {v: i for i, v in enumerate(order)}
    return order[(pos[y] + pos[x] - step) % p]


def verify(module, epochs, bound):
    m = verify_matmul(module, epochs, bound, compute_modes=("dsr",))
    a, b, op, out = m["nodes"]
    d = op["dataflow"]
    check(
        all(n.get("dtype") == "f16" for n in (a, b, op)),
        "two-hop binary16 inputs/result",
    )
    check(
        set(d)
        == {
            "rows",
            "cols",
            "exchange",
            "initial_align",
            "reduce",
            "overlap",
            "fp",
            "compute",
        },
        "two-hop policy fields",
    )
    check(
        d["exchange"] == "two_hop"
        and d["initial_align"] == "bidirectional"
        and d["reduce"] == "local"
        and d["overlap"] == "double_buffer",
        "two-hop communication contract",
    )
    out["dtype"] = "f16"
    m["profile"] = "mesh_twohop.v1"
    plan(m)
    return m


def plan(m, partitions=1):
    a, b, op, out = m["nodes"]
    d = op["dataflow"]
    p = d["rows"]
    order = cycle(p)
    check(partitions == 1 and d["cols"] == p, "two-hop square ownership")
    rows, k = a["shape"]
    cols = b["shape"][1]
    check(all(v % p == 0 for v in (rows, k, cols)), "two-hop divisible matrix blocks")
    mt, kt, nt = rows // p, k // p, cols // p
    check(
        mt * kt % 2 == 0 and kt * nt % 2 == 0,
        "two-hop packed half communication requires even tile extents",
    )
    check(
        max(mt * kt, kt * nt, p * mt * nt) <= 32767, "two-hop signed DSD/offset bounds"
    )
    mode = m.get("instrumentation", "sampled")
    check(mode in ("sampled", "counters"), "two-hop instrumentation mode")
    memory = dict(
        resident_half_buffers=2 * (2 * mt * kt + 2 * kt * nt + mt * nt),
        half_prefix_history=2 * p * mt * nt if mode == "sampled" else 2,
        witnesses_timestamps_counters=8 * p + 12 * p + 128,
        sdk_code_tasks_stack_reserve=16384,
    )
    check(sum(memory.values()) <= 49152, "two-hop PE memory budget")
    return dict(
        profile="mesh_twohop.v1",
        matrix_rows=rows,
        matrix_k=k,
        matrix_cols=cols,
        P=p,
        Mt=mt,
        Kt=kt,
        Nt=nt,
        epochs=m["epochs"],
        dtype="f16",
        instrumentation=mode,
        compute="dsr",
        cycle=order,
        stages=[
            "host permutes W block rows; X unshifted",
            "bidirectional X alignment on device",
            "overlap next X/W communication with current DSR FMA",
            "join both axis completions before buffer swap",
        ],
        k_block_rule="cycle[(position(y)+position(x)-round)%P]",
        memory_per_pe=memory,
        resources=dict(
            colors=[1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16],
            input_queues=[2, 3, 4, 5],
            output_queues=[2, 3, 4, 5],
            microthreads=[1, 2, 3, 4],
            local_tasks=[22, 23, 24, 25, 26],
            compute_dsr=1,
            communication_dsrs=[2, 3, 4, 5, 6, 7],
            host_bindings="fixed X_0/W_0/res pointer cells; working send/receive pointers separate",
            join="X send/receive -> x_finish; Y send/receive -> y_finish; join unblocks next_step after compute activation",
        ),
        numeric_policy="binary16 fused round-to-nearest-even per update; relaxed block K ordering",
        tile_orders=dict(X="column-major", W="row-major", result="column-major"),
        nodes=[
            dict(id=f"p{x}_{y}", tile=[x, y], place=[4 + x, 1 + y])
            for y in range(p)
            for x in range(p)
        ],
    )


def pack(a, b, p):
    import numpy as np

    aa = pack_tiles(a, p, p, "F")
    wb = pack_tiles(b, p, p, "C")
    bb = np.asarray([[wb[block_index(p, y, x), x] for x in range(p)] for y in range(p)])
    return aa, bb


def generate(s, dest):
    runtime = source_path("runtime")
    dest = Path(dest)
    for output, source in {
        "pe.csl": "twohop_pe.csl",
        "layout.csl": "twohop_layout.csl",
        "twohop_comm.csl": "twohop_comm.csl",
        "twohop_routes.csl": "twohop_routes.csl",
    }.items():
        text = (runtime / source).read_text()
        if output == "pe.csl" and s["instrumentation"] == "counters":
            store = "        const target=@increment_dsd_offset(hls_history_dsd,step*Mt*Nt,f16);\n        @fmovh(target,res_dsd);hls_progress[0]+=1;"
            check(text.count(store) == 1, "two-hop history store site")
            text = text.replace(store, "        hls_progress[0]+=1;")
            text = text.replace(
                "var hls_history=@zeros([P*Mt*Nt]f16);",
                "var hls_history=@zeros([1]f16);",
            )
            text = text.replace(
                "const hls_history_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{Mt*Nt}->hls_history[i]});",
                "",
            )
        (dest / output).write_text(text)
