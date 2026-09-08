"""Build source-backed HLS projection/residual/RMS with an original-input math gate."""

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
from application_gate import seal
from projection_residual_rms_fixtures import batches, check


def source(m=64, n=64, p=8):
    return f"""#include "spatial.hpp"
void design(){{
 auto x=spatial::input<{m},{n},spatial::f16>("activation",0.125);
 auto w=spatial::input<{n},{n},spatial::f16>("weight",0.125);
 auto r=spatial::input<{m},{n},spatial::f16>("residual",0.5);
 auto g=spatial::input<1,{n},spatial::f16>("gamma",1.5);
 #pragma csl dataflow rows={p} cols={p} exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto projected=spatial::matmul(x,w);
 #pragma csl dataflow rows={p} cols={p} partition=tiles compute=dsr fp=relaxed
 auto z=spatial::add(projected,r);
 #pragma csl dataflow rows={p} cols={p} partition=tiles reduce=bidirectional_chain weights=feature_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto normalized=spatial::rmsnorm(z,g,0.000001);
 spatial::output("output",normalized);
}}
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--p", type=int, default=8)
    p.add_argument(
        "--instrumentation", choices=["sampled", "counters"], default="sampled"
    )
    a = p.parse_args()
    folder = (
        ROOT
        / "benchmarks/waferllm"
        / (
            f"projection_residual_rms_{a.m}x{a.n}_{a.p}x{a.p}"
            + ("_counters" if a.instrumentation == "counters" else "")
        )
    )
    text = source(a.m, a.n, a.p)
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
                    origins=["Prefill/src/prefill.csl h1_matmul/z_add/rmsnorm_z"],
                    source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                    contract="Supplied activation, square projection weights, residual and feature gamma; resident projection/add/RMS composition under development. No full Prefill or hardware qualification.",
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
        batches=batches(a.m, a.n),
        instrumentation=a.instrumentation,
        sdk_options=dict(suppress_trace=True, num_threads=8, dump_core=True),
    )
    seal(
        dest,
        ROOT / "tests/support/projection_residual_rms_fixtures.py",
        lambda b, o: check(a.m, a.n, 1e-6, b, o),
        dict(M=a.m, N=a.n),
    )
    shutil.copyfile(
        ROOT / "experiments/execute_frozen_bundle.py", dest / "sdk-execution-driver.py"
    )
    print(dest.relative_to(ROOT), flush=True)
