"""Build the shared whole-graph HLS and seal real C++/target stage gates."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import datetime, sys
from pathlib import Path

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib"), str(ROOT / "experiments")]
from compile import build
from projected_cache_fixtures import batches
from projected_cache_gate import seal_native, seal_target


def main():
    project = ROOT / "benchmarks/inference/waferllm/projected_cache_attention_3x256x512_8x8"
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
