"""Build typed shared-normalization projection fan-out using existing HLS ops."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, shutil, sys
from pathlib import Path

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from compile import build
from normalized_fanout_fixtures import batches

p = argparse.ArgumentParser()
p.add_argument("--sdk-threads", type=int, default=8)
p.add_argument("--m", type=int, default=64)
p.add_argument("--n", type=int, default=128)
p.add_argument("--p", type=int, default=8)
p.add_argument("--projections", type=int, choices=(2, 3), default=3)
p.add_argument("--instrumentation", choices=["sampled", "counters"], default="sampled")
a = p.parse_args()
f = (
    ROOT
    / "benchmarks/waferllm"
    / f"normalized_fanout{a.projections}_{a.m}x{a.n}_{a.p}x{a.p}"
)
f.mkdir(exist_ok=True)
source = f"""#include "spatial.hpp"
void design() {{
 auto x=spatial::input<{a.m},{a.n},spatial::f16>("x");
 auto w=spatial::input<1,{a.n},spatial::f16>("w");
 #pragma csl dataflow rows={a.p} cols={a.p} partition=tiles reduce=bidirectional_chain weights=feature_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto normalized=spatial::rmsnorm(x,w,0.000001);
"""
for i in range(a.projections):
    source += f""" auto weight{i}=spatial::input<{a.n},{a.n},spatial::f16>("weight{i}");
 #pragma csl dataflow rows={a.p} cols={a.p} exchange=two_hop initial_align=forward reduce=local overlap=double_buffer fp=relaxed compute=dsr
 auto projection{i}=spatial::matmul(normalized,weight{i});
 spatial::output("projection{i}",projection{i});
"""
source += "}\n"
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
                    "Prefill/src/prefill.csl rmsnorm_x/xq_matmul/xk_matmul/xv_matmul"
                ],
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                status="development",
                partitions=1,
                contract="Shared RMS normalization and spatial alignment, sequential square projections preserving live double-buffer ownership. No intermediate host transfer or complete inference claim.",
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
    batches=batches(a.m, a.n, a.projections),
    instrumentation=a.instrumentation,
    sdk_options=dict(suppress_trace=True, num_threads=a.sdk_threads, dump_core=True),
)
shutil.copyfile(
    ROOT / "experiments/execute_frozen_bundle.py", d / "sdk-execution-driver.py"
)
print(d.relative_to(ROOT), flush=True)
