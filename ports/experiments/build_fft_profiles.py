"""Author fresh explicit FFT profiles and freeze native-checked builds."""

import argparse, datetime, json, shutil, sys, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "toolchain"), str(ROOT)]
from compile import build
from fft_fixtures import batches

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, required=True)
p.add_argument("--mesh", type=int, required=True)
p.add_argument("--direction", choices=["forward", "inverse"], default="forward")
p.add_argument("--norm", choices=["backward", "ortho", "forward"], default="backward")
p.add_argument("--input-batches", type=Path)
p.add_argument("--instrumentation", choices=["sampled", "counters"], default="counters")
p.add_argument(
    "--result-layout",
    choices=["input_layout", "transposed_pencils"],
    default="input_layout",
)
a = p.parse_args()
name = f"fft3d_{a.n}_{a.mesh}x{a.mesh}_{a.direction}_{a.norm}"
if a.result_layout != "input_layout":
    name += "_transposed"
folder = ROOT / "projects/sdk_examples" / name
source = f"""#include "spatial.hpp"
void design() {{
 auto x=spatial::input<{a.n*a.n},{2*a.n}>("x");
 #pragma csl dataflow rows={a.mesh} cols={a.mesh} partition=pencils exchange=sdk_transpose compute=sdk_fft result={a.result_layout} fp=relaxed
 auto spectrum=spatial::fft3d<{a.n},spatial::fft_direction::{a.direction},spatial::fft_norm::{a.norm}>(x);
 spatial::output("spectrum",spectrum);
}}
"""
folder.mkdir(exist_ok=True)
if (folder / "hls.cpp").exists():
    assert (folder / "hls.cpp").read_text() == source, "preserve existing source"
else:
    (folder / "hls.cpp").write_text(source)
    (folder / "PORT.json").write_text(
        json.dumps(
            dict(
                project="sdk_examples",
                kernel=name,
                origins=["benchmarks/fft-3d/layout.csl"],
                fixture=f"distributed_fft:{a.n}:{a.direction}:{a.norm}",
                partitions=1,
                status="development",
                contract=f"SDK C2C {a.n}^3 {a.direction}/{a.norm} pencils; qualification pending",
            ),
            indent=2,
        )
        + "\n"
    )
b = json.loads(a.input_batches.read_text()) if a.input_batches else batches(a.n)
dest = folder / (
    "run-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
)
build(
    folder / "hls.cpp",
    dest,
    instrumentation=a.instrumentation,
    epochs=len(b),
    bound=max(
        8,
        math.ceil(
            max(
                abs(float(v))
                for batch in b
                for values in batch.values()
                for v in values
            )
        ),
    ),
    batches=b,
    sdk_options=dict(suppress_trace=True, num_threads=8, dump_core=True),
)
shutil.copyfile(
    ROOT / "experiments/execute_frozen_bundle.py", dest / "sdk-execution-driver.py"
)
print(dest.relative_to(ROOT), flush=True)
