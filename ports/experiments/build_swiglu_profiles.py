"""Build source-inspired gated activation with typed last-use buffer reuse."""

import argparse, datetime, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from compile import build
from swiglu_fixtures import batches

p = argparse.ArgumentParser()
p.add_argument("--m", type=int, default=64)
p.add_argument("--n", type=int, default=256)
p.add_argument("--rows", type=int, default=8)
p.add_argument("--cols", type=int, default=8)
p.add_argument("--instrumentation", choices=["sampled", "counters"], default="sampled")
a = p.parse_args()
f = ROOT / "projects/waferllm" / f"swiglu_{a.m}x{a.n}_{a.rows}x{a.cols}"
f.mkdir(exist_ok=True)
policy = f"rows={a.rows} cols={a.cols} partition=tiles math=sdk_half compute=dsr elementwise=map fp=relaxed"
source = f"""#include "spatial.hpp"
void design() {{
 auto up=spatial::input<{a.m},{a.n},spatial::f16>("up");
 auto gate=spatial::input<{a.m},{a.n},spatial::f16>("gate");
 #pragma csl dataflow {policy}
 auto activated=spatial::silu(gate);
 #pragma csl dataflow {policy}
 auto gated=spatial::multiply(up,activated);
 spatial::output("gated",gated);
}}
"""
if (f / "hls.cpp").exists():
    assert (f / "hls.cpp").read_text() == source
else:
    (f / "hls.cpp").write_text(source)
    (f / "PORT.json").write_text(
        json.dumps(
            dict(
                project="waferllm",
                kernel=f.name,
                origins=["Prefill/src/prefill.csl silu_kernel/z3_comp"],
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                contract="Gated SiLU of precomputed up/gate tensors; SDK half exp, CSL map and DSR multiply, finite magnitude <=8, stored half activation. No projection/full MLP/inference claim.",
                status="development",
                partitions=1,
            ),
            indent=2,
        )
        + "\n"
    )
d = f / (
    "run-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
)
build(
    f / "hls.cpp",
    d,
    epochs=6,
    bound=8,
    batches=batches(a.m, a.n),
    instrumentation=a.instrumentation,
    sdk_options=dict(suppress_trace=True, num_threads=8, dump_core=True),
)
shutil.copyfile(
    ROOT / "experiments/execute_frozen_bundle.py", d / "sdk-execution-driver.py"
)
print(d.relative_to(ROOT), flush=True)
