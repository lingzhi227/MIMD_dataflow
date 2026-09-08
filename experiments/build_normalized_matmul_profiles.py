"""Build a resident typed normalization/contraction subgraph, not host staging."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, shutil, sys
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from compile import build
from normalized_matmul_fixtures import batches

p = argparse.ArgumentParser()
p.add_argument("--m", type=int, default=64)
p.add_argument("--n", type=int, default=128)
p.add_argument("--p", type=int, default=8)
p.add_argument("--instrumentation", choices=["sampled", "counters"], default="sampled")
a = p.parse_args()
f = ROOT / "benchmarks/waferllm" / f"normalized_matmul_{a.m}x{a.n}_{a.p}x{a.p}"
f.mkdir(exist_ok=True)
source = f"""#include "spatial.hpp"
void design() {{
 auto x=spatial::input<{a.m},{a.n},spatial::f16>("x");
 auto w=spatial::input<1,{a.n},spatial::f16>("w");
 #pragma csl dataflow rows={a.p} cols={a.p} partition=tiles reduce=bidirectional_chain weights=feature_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto normalized=spatial::rmsnorm(x,w,0.000001);
 auto q=spatial::input<{a.n},{a.n},spatial::f16>("q");
 #pragma csl dataflow rows={a.p} cols={a.p} exchange=two_hop initial_align=forward reduce=local overlap=double_buffer fp=relaxed compute=dsr
 auto projected=spatial::matmul(normalized,q);
 spatial::output("projected",projected);
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
                    "Prefill/src/prefill.csl rmsnorm_x/xq_matmul/matmul_compute",
                    "Prefill/src/comm_lib/comm_pe.csl",
                ],
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                contract="Resident RMSNorm -> projection, source-compatible forward two-hop communication and explicit corrected per-row scale; no intermediate host transfer.",
                status="development",
                partitions=1,
            ),
            indent=2,
        )
        + "\n"
    )
b = batches(a.m, a.n)

d = f / (
    "run-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
)
build(
    f / "hls.cpp",
    d,
    epochs=6,
    bound=1,
    batches=b,
    instrumentation=a.instrumentation,
    sdk_options=dict(suppress_trace=True, num_threads=8, dump_core=True),
)
shutil.copyfile(
    ROOT / "experiments/execute_frozen_bundle.py", d / "sdk-execution-driver.py"
)
print(d.relative_to(ROOT), flush=True)
