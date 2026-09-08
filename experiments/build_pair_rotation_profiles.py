"""Author explicit token-dependent or broadcast pair-rotation HLS dataflow."""

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
from pair_rotation_fixtures import batches

p = argparse.ArgumentParser()
p.add_argument("--m", type=int, default=64)
p.add_argument("--n", type=int, default=128)
p.add_argument("--rows", type=int, default=8)
p.add_argument("--cols", type=int, default=8)
p.add_argument("--broadcast", action="store_true")
p.add_argument("--order", choices=["even_odd", "odd_even"], default="even_odd")
p.add_argument("--instrumentation", choices=["sampled", "counters"], default="sampled")
a = p.parse_args()
f = (
    ROOT
    / "benchmarks/waferllm"
    / f"pair_rotation_{a.m}x{a.n}_{a.rows}x{a.cols}_{'broadcast' if a.broadcast else 'token'}_{a.order}"
)
f.mkdir(exist_ok=True)
source = f"""#include "spatial.hpp"
void design() {{
 auto x=spatial::input<{a.m},{a.n},spatial::f16>("x");
 auto cosine=spatial::input<{1 if a.broadcast else a.m},{a.n//2},spatial::f16>("cosine");
 auto sine=spatial::input<{1 if a.broadcast else a.m},{a.n//2},spatial::f16>("sine");
 #pragma csl dataflow rows={a.rows} cols={a.cols} partition=tiles coefficients={'feature_pairs' if a.broadcast else 'per_token'} compute=dsd fp=relaxed
 auto rotated=spatial::rotate_pairs<spatial::pair_order::{a.order}>(x,cosine,sine);
 spatial::output("rotated",rotated);
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
                origins=["Prefill/src/prefill.csl xq_rope/xk_rope"],
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                status="development",
                partitions=1,
                contract="Explicit adjacent-pair input order and coefficient row ownership. Token-row-sized DSD scratch; finite half magnitude<=8. Not a full model RoPE/inference reproduction.",
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
    batches=batches(a.m, a.n, a.broadcast, a.order),
    instrumentation=a.instrumentation,
    sdk_options=dict(suppress_trace=True, num_threads=8, dump_core=True),
)
shutil.copyfile(
    ROOT / "experiments/execute_frozen_bundle.py", d / "sdk-execution-driver.py"
)
print(d.relative_to(ROOT), flush=True)
