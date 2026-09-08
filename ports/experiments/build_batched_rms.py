"""Build a fresh standard HLS bundle and seal independent RMS native accuracy."""

import argparse, datetime, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from compile import build
from batched_rms_fixtures import batches, check
from application_gate import seal

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("project", type=Path)
    a = p.parse_args()
    out = a.project / (
        "run-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    print(out, flush=True)
    root = build(
        a.project / "hls.cpp",
        out,
        epochs=8,
        bound=2,
        batches=batches(3, 512),
        instrumentation="sampled",
        sdk_options=dict(suppress_trace=True, num_threads=8, dump_core=True),
    )
    seal(
        root,
        ROOT / "rms_fixtures.py",
        lambda b, o: check(3, 512, b, o),
        dict(B=3, N=512),
    )
    print("PASS native and original-domain RMS gate", flush=True)
