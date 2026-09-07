"""Build standard stable softmax with source-failure and unequal-tile regressions."""

import argparse, datetime, json, math, shutil, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from softmax_fixtures import batches as fixtures
from compile import build

p = argparse.ArgumentParser()
p.add_argument("--m", type=int, default=64)
p.add_argument("--n", type=int, default=128)
p.add_argument("--rows", type=int, default=8)
p.add_argument("--cols", type=int, default=8)
p.add_argument("--instrumentation", choices=["sampled", "counters"], default="sampled")
p.add_argument("--elementwise", choices=["scalar", "map"], default="scalar")
a = p.parse_args()
m, n = a.m, a.n
folder = ROOT / "projects/waferllm" / f"softmax_{m}x{n}_{a.rows}x{a.cols}"
if a.elementwise == "map":
    folder = folder.with_name(folder.name + "_map")
folder.mkdir(exist_ok=True)
source = f"""#include "spatial.hpp"
void design() {{
 auto x=spatial::input<{m},{n},spatial::f16>("x");
 #pragma csl dataflow rows={a.rows} cols={a.cols} partition=tiles reduce=max_sum accumulation=f16 math=sdk_half compute=dsr fp=relaxed{ " elementwise=map" if a.elementwise == "map" else ""}
 auto probability=spatial::softmax(x,0.125);
 spatial::output("probability",probability);
}}
"""
if (folder / "hls.cpp").exists():
    assert (folder / "hls.cpp").read_text() == source
else:
    (folder / "hls.cpp").write_text(source)
    (folder / "PORT.json").write_text(
        json.dumps(
            dict(
                project="waferllm",
                kernel=folder.name,
                origins=[
                    "Prefill/src/prefill.csl softmax_score",
                    "Prefill/src/comm_lib/comm_pe.csl",
                ],
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                contract="Stable row softmax; correct all-negative maximum initialization, SDK exp and source-inspired max/sum row chains.",
                status="development",
                partitions=1,
            ),
            indent=2,
        )
        + "\n"
    )
b = fixtures(m, n)
dest = folder / (
    "run-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
)
build(
    folder / "hls.cpp",
    dest,
    epochs=6,
    bound=1024,
    batches=b,
    instrumentation=a.instrumentation,
    sdk_options=dict(suppress_trace=True, num_threads=8, dump_core=True),
)
shutil.copyfile(
    ROOT / "experiments/execute_frozen_bundle.py", dest / "sdk-execution-driver.py"
)
print(dest.relative_to(ROOT), flush=True)
