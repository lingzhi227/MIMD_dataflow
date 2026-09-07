"""Build source-scheduled QK-transpose through the common HLS pipeline."""

import argparse, datetime, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from compile import build
from device_matmul_fixtures import batches

p = argparse.ArgumentParser()
p.add_argument("--m", type=int, default=64)
p.add_argument("--n", type=int, default=128)
p.add_argument("--p", type=int, default=8)
p.add_argument("--instrumentation", choices=["sampled", "counters"], default="sampled")
a = p.parse_args()
f = ROOT / "projects/waferllm" / f"device_matmul_{a.m}x{a.n}_{a.p}x{a.p}"
f.mkdir(exist_ok=True)
source = f"""#include "spatial.hpp"
void design() {{
 auto a=spatial::input<{a.m},{a.m},spatial::f16>("a");
 auto b=spatial::input<{a.m},{a.n},spatial::f16>("b");
 #pragma csl dataflow rows={a.p} cols={a.p} exchange=two_hop initial_align=both_axes reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto product=spatial::matmul(a,b);
 spatial::output("product",product);
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
                origins=[
                    "Prefill/src/prefill.csl output_matmul with device alignment/strided layout"
                ],
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                status="development",
                partitions=1,
                contract="Logical column-major tiles, both alignments on device, strided RHS without a local transpose. Source-derived matrix contraction, no complete attention claim.",
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
    bound=1,
    batches=batches(a.m, a.n),
    instrumentation=a.instrumentation,
    sdk_options=dict(suppress_trace=True, num_threads=8, dump_core=True),
)
shutil.copyfile(
    ROOT / "experiments/execute_frozen_bundle.py", d / "sdk-execution-driver.py"
)
print(d.relative_to(ROOT), flush=True)
