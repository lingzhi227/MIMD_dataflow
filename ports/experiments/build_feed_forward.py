"""Build an ordinary typed normalized feed-forward residual with a native math gate."""

import argparse, datetime, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from compile import build
from feed_forward_fixtures import batches


def source(m=64, n=64, f=256, p=8, all_blocked=True):
    proj = f"#pragma csl dataflow rows={p} cols={p} exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed"
    point = f"#pragma csl dataflow rows={p} cols={p} partition=tiles math=sdk_half compute=dsr elementwise=map fp=relaxed"
    upper_policy = proj + (" accumulation=block_f32" if all_blocked else "")
    up_call = (
        f"spatial::matmul_blocked<spatial::f16,spatial::scalar>(x,u,{n//p})"
        if all_blocked
        else "spatial::matmul(x,u)"
    )
    gate_call = (
        f"spatial::matmul_blocked<spatial::f16,spatial::scalar>(x,g,{n//p})"
        if all_blocked
        else "spatial::matmul(x,g)"
    )
    return f"""#include "spatial.hpp"
void design(){{
 auto z=spatial::input<{m},{n},spatial::f16>("z",0.125);
 auto gamma=spatial::input<1,{n},spatial::f16>("gamma",1.5);
 auto u=spatial::input<{n},{f},spatial::f16>("up_weight",0.0078125);
 auto g=spatial::input<{n},{f},spatial::f16>("gate_weight",0.0078125);
 auto d=spatial::input<{f},{n},spatial::f16>("down_weight",0.0078125);
 #pragma csl dataflow rows={p} cols={p} partition=tiles reduce=bidirectional_chain weights=feature_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto x=spatial::rmsnorm(z,gamma,0.000001);
 {upper_policy}
 auto up={up_call};
 {upper_policy}
 auto gate={gate_call};
 {point}
 auto act=spatial::silu(gate);
 {point}
 auto hidden=spatial::multiply(up,act);
 {proj} accumulation=block_f32
 auto delta=spatial::matmul_blocked<spatial::f16,spatial::scalar>(hidden,d,{f//p});
 #pragma csl dataflow rows={p} cols={p} partition=tiles compute=dsr fp=relaxed
 auto result=spatial::add(z,delta);
 spatial::output("output",result);
}}
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--instrumentation", choices=["sampled", "counters"], default="sampled"
    )
    ap.add_argument("--accumulation", choices=["all", "down"], default="all")
    ap.add_argument(
        "--geometry",
        type=int,
        nargs=4,
        default=(64, 64, 256, 8),
        metavar=("M", "N", "F", "P"),
    )
    a = ap.parse_args()
    m, n, f, p = a.geometry
    folder = (
        ROOT
        / "projects/waferllm"
        / (
            f"feed_forward_{m}x{n}x{f}_{p}x{p}"
            + ("_all_blocked" if a.accumulation == "all" else "")
            + ("_counters" if a.instrumentation == "counters" else "")
        )
    )
    text = source(m, n, f, p, all_blocked=a.accumulation == "all")
    if folder.exists():
        assert (folder / "hls.cpp").read_text() == text
    else:
        folder.mkdir()
        (folder / "hls.cpp").write_text(text)
        (folder / "PORT.json").write_text(
            json.dumps(
                dict(
                    project="waferllm",
                    kernel=folder.name,
                    status="development",
                    partitions=1,
                    source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                    origins=[
                        "Prefill/src/prefill.csl rmsnorm_z/z1_matmul/z2_matmul/z3_comp/h2_matmul/add_result"
                    ],
                    contract="Supplied Z normalized feed-forward residual, explicit block-f32 down. Development; not full Prefill/Decode.",
                ),
                indent=2,
            )
            + "\n"
        )
    dest = folder / (
        "run-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    build(
        folder / "hls.cpp",
        dest,
        epochs=8,
        bound=2,
        batches=batches(m, n, f, p),
        instrumentation=a.instrumentation,
        sdk_options=dict(suppress_trace=True, num_threads=8, dump_core=True),
    )
    from feed_forward_gate import seal_native, seal_target

    seal_native(dest)
    seal_target(dest)
    shutil.copyfile(
        ROOT / "experiments/execute_frozen_bundle.py", dest / "sdk-execution-driver.py"
    )
    print(dest.relative_to(ROOT), flush=True)
