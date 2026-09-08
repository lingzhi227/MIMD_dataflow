"""Build Decode-layout pair rotation through shared HLS and independent native gates."""

import datetime, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain"), str(ROOT / "experiments")]
from compile import build
from pair_rotation_fixtures import batches, check
from application_gate import seal


def main():
    project = ROOT / "projects/waferllm/batched_pair_rotation_5x1024_8x8_x"
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
        ROOT / "pair_rotation_fixtures.py",
        lambda b, o: check(5, 1024, True, "odd_even", b, o),
        dict(M=5, N=1024, broadcast=True, order="odd_even"),
    )
    print(root.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    main()
