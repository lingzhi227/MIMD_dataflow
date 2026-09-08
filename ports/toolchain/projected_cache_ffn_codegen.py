"""Emit a caller-owned resident region and typed physical ports from its plan."""

from pathlib import Path
import numpy as np
from frontend import check
from rms_statistic_ir import StatisticType, verify_connection


def extents(s):
    ports = {
        k: v // 2
        for k, v in s["numeric_allocations"].items()
        if not k.startswith(("collective_", "max_", "mean_")) and "_block_" not in k
    }
    ports.update(progress=1, stages=8, timing=6, queues=2, attention_progress=11)
    return ports


def parameters(s):
    a = s["attention"]
    tail = s["graph"]["ffn"]["nodes"]
    v = {
        k: a[k]
        for k in ("P", "B", "Nt", "St", "score_block", "value_block", "output_block")
    }
    v.update(dict(zip(("q_block", "k_block", "v_block"), a["projection_blocks"])))
    v.update(
        Ft=s["Ft"],
        capacity=s["collective_capacity"],
        sampled=1,
        scale_bits=int(np.float16(a["scale"]).view(np.uint16)),
        epsilon_bits=int(np.float16(a["epsilon"]).view(np.uint16)),
        ffn_epsilon_bits=int(np.float16(tail[5]["epsilon"]).view(np.uint16)),
        up_block=tail[6]["block_size"],
        gate_block=tail[7]["block_size"],
        down_block=tail[10]["block_size"],
    )
    for k, value in v.items():
        check(
            type(value) is int
            and 0 <= value <= (65535 if k.endswith("_bits") else 32767),
            "typed CSL parameter " + k,
        )
    return v


def generate(s, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    check(not list(dest.glob("*.csl")), "fresh composed CSL output")
    # This consumer template expects the reduced mean, never sum/N.
    produced = StatisticType(**s["boundary"]["statistic_type"])
    verify_connection(
        produced, StatisticType("mean", s["attention"]["N"], s["attention"]["N"])
    )
    v = parameters(s)
    p = v["P"]
    rt = Path(__file__).parent / "runtime"
    pe = (rt / "projected_cache_ffn_pe.csl").read_text()
    layout = f"""const P:i16={p};
const memcpy=@import_module("<memcpy/get_params>",.{{.width=P,.height=P}});
const c2d=@import_module("<collectives_2d/params>");
layout {{
 @set_rectangle(P,P);
 for(@range(i16,P))|y|{{for(@range(i16,P))|x|{{
 @set_tile_code(x,y,"pe.csl",.{{.memcpy_params=memcpy.get_params(x),
"""
    layout += "".join(f".{k}={value}," for k, value in v.items())
    layout += """
 .c2d_params=c2d.get_params(@as(u16,x),@as(u16,y),.{
 .x_colors=.{@get_color(0),@get_color(1)},.x_entrypoints=.{@get_local_task_id(14),@get_local_task_id(15)},
 .y_colors=.{@get_color(4),@get_color(5)},.y_entrypoints=.{@get_local_task_id(16),@get_local_task_id(17)}
 })});
 }}
"""
    exported = []
    meta = {"progress", "stages", "timing", "queues", "attention_progress"}
    for name in extents(s):
        dtype = "u16" if name in meta else "f16"
        variable = (
            "attention.progress"
            if name == "attention_progress"
            else (
                name if name.startswith("ffn_") or name in meta else "attention." + name
            )
        )
        pe += f"\nvar export_{name}:[*]{dtype}=&{variable};\n"
        exported.append(f' @export_symbol(export_{name},"{name}");')
        layout += f' @export_name("{name}",[*]{dtype},true);\n'
    pe += (
        "\ncomptime {\n"
        + "\n".join(exported)
        + "\n @export_symbol(init_task);@export_symbol(hls_main);\n}\n"
    )
    layout += (
        ' @export_name("init_task",fn()void);@export_name("hls_main",fn()void);\n}\n'
    )
    (dest / "pe.csl").write_text(pe)
    (dest / "layout.csl").write_text(layout)
    for name in (
        "projected_cache_region.csl",
        "batched_rms_local.csl",
        "batched_matmul_blocked.csl",
        "batched_matmul_local.csl",
        "batched_pair_rotation_local.csl",
        "batched_softmax_local.csl",
        "sdk_axis_reduce.csl",
        "sdk_axis_max.csl",
        "sdk_axis_mean.csl",
        "sdk_stable_silu.csl",
    ):
        (dest / name).write_bytes((rt / name).read_bytes())
