"""Build Decode-layout pair rotation through shared HLS and independent native gates."""

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
from pair_rotation_fixtures import batches, check
from application_gate import seal


def main():
    project = ROOT / "benchmarks/inference/waferllm/batched_pair_rotation_5x1024_8x8_x"
    root = project / (
        "run-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    build(
        project / "hls.cpp",
        root,
        epochs=6,
        bound=8,
        batches=batches(5, 1024, True, "odd_even"),
        instrumentation="sampled",
        sdk_options=dict(suppress_trace=True, num_threads=8, dump_core=False),
    )
    seal(
        root,
        ROOT / "tests/support/pair_rotation_fixtures.py",
        lambda b, o: check(5, 1024, True, "odd_even", b, o),
        dict(M=5, N=1024, broadcast=True, order="odd_even"),
    )
    print(root.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
