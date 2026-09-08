"""Build the shared whole-graph HLS and seal real C++/target stage gates."""

import datetime, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain"), str(ROOT / "experiments")]
from compile import build
from projected_cache_fixtures import batches
from projected_cache_gate import seal_native, seal_target


def main():
    project = ROOT / "projects/waferllm/projected_cache_attention_3x256x512_8x8"
    dest = project / (
        "run-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    print(dest.relative_to(ROOT), flush=True)
    build(
        project / "hls.cpp",
        dest,
        epochs=8,
        bound=2,
        batches=batches(3, 256, 512),
        instrumentation="sampled",
        sdk_options=dict(suppress_trace=True, num_threads=8, dump_core=False),
    )
    seal_native(dest)
    seal_target(dest)
    print("PROJECTED CACHE PREFLIGHT PASS", dest, flush=True)


if __name__ == "__main__":
    main()
