"""Build the supplied-attention projection, residual and normalized FFN tail."""

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
from build_feed_forward import source as ff_source
from prefill_tail_fixtures import batches
from compile import build


def source(m=64, n=64, f=256, p=8):
    text = ff_source(m, n, f, p)
    old = f' auto z=spatial::input<{m},{n},spatial::f16>("z",0.125);'
    text = text.replace(
        old,
        f""" auto attention=spatial::input<{m},{n},spatial::f16>("attention",0.125);
 auto output_weight=spatial::input<{n},{n},spatial::f16>("output_weight",0.0078125);
 auto residual=spatial::input<{m},{n},spatial::f16>("residual",0.125);""",
    )
    marker = f" #pragma csl dataflow rows={p} cols={p} partition=tiles reduce=bidirectional_chain"
    pos = text.index(marker)
    text = (
        text[:pos]
        + f""" #pragma csl dataflow rows={p} cols={p} exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed
 auto projection=spatial::matmul(attention,output_weight);
 #pragma csl dataflow rows={p} cols={p} partition=tiles compute=dsr fp=relaxed
 auto z=spatial::add(projection,residual);
"""
        + text[pos:]
    )
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--instrumentation", choices=["sampled", "counters"], default="counters"
    )
    ap.add_argument("--geometry", type=int, nargs=4, default=(64, 64, 256, 8))
    a = ap.parse_args()
    m, n, f, p = a.geometry
    folder = (
        ROOT
        / "benchmarks/waferllm"
        / f"prefill_tail_{m}x{n}x{f}_{p}x{p}_{a.instrumentation}"
    )
    text = source(m, n, f, p)
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
                        "Prefill/src/prefill.csl h1_matmul/z_add/rmsnorm_z/z1_matmul/z2_matmul/z3_comp/h2_matmul/add_result"
                    ],
                    contract="Supplied attention output tail; half output projection and block-f32 MLP projections; not full Prefill or Decode.",
                ),
                indent=2,
            )
            + "\n"
        )
    dest = folder / (
        "run-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    print(dest.relative_to(ROOT), flush=True)
    build(
        folder / "hls.cpp",
        dest,
        epochs=8,
        bound=2,
        batches=batches(m, n, f, p),
        instrumentation=a.instrumentation,
        sdk_options=dict(suppress_trace=True, num_threads=8, dump_core=True),
    )
    from prefill_tail_gate import seal_native, seal_target

    seal_native(dest)
    seal_target(dest)
    shutil.copyfile(
        ROOT / "experiments/execute_frozen_bundle.py", dest / "sdk-execution-driver.py"
    )
