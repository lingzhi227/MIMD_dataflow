"""Build source-expressed distributed RMSNorm with adversarial axis fixtures."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, sys, shutil
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from compile import build

p = argparse.ArgumentParser()
p.add_argument("--m", type=int, default=64)
p.add_argument("--n", type=int, default=128)
p.add_argument("--rows", type=int, default=8)
p.add_argument("--cols", type=int, default=8)
p.add_argument("--instrumentation", choices=["sampled", "counters"], default="sampled")
a = p.parse_args()
folder = ROOT / "benchmarks/waferllm" / f"rmsnorm_{a.m}x{a.n}_{a.rows}x{a.cols}"
folder.mkdir(exist_ok=True)
source = f"""#include "spatial.hpp"
void design() {{
 auto x=spatial::input<{a.m},{a.n},spatial::f16>("x");
 auto w=spatial::input<1,{a.n},spatial::f16>("w");
 #pragma csl dataflow rows={a.rows} cols={a.cols} partition=tiles reduce=bidirectional_chain weights=feature_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto normalized=spatial::rmsnorm(x,w,0.000001);
 spatial::output("normalized",normalized);
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
                    "Prefill/src/prefill.csl rmsnorm_x",
                    "Prefill/src/comm_lib/comm_pe.csl mv_allreduce_add_x",
                ],
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                contract="Standard row RMSNorm, explicitly correcting original feature-indexed scale and host weight ownership; source communication and half arithmetic strategy retained.",
                status="development",
                partitions=1,
            ),
            indent=2,
        )
        + "\n"
    )
sys.path.insert(0, str(ROOT))
from rms_fixtures import batches

b = batches(a.m, a.n)

dest = folder / (
    "run-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
)
build(
    folder / "hls.cpp",
    dest,
    epochs=len(b),
    bound=1,
    batches=b,
    instrumentation=a.instrumentation,
    sdk_options=dict(suppress_trace=True, num_threads=8, dump_core=True),
)
shutil.copyfile(
    ROOT / "experiments/execute_frozen_bundle.py", dest / "sdk-execution-driver.py"
)
print(dest.relative_to(ROOT), flush=True)
