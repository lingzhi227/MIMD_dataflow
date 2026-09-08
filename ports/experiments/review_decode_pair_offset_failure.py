"""Retain actual unmodified Decode mismatch and the independently executed DSD cause."""

import json, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
from pair_rotation_fixtures import check

h = (
    ROOT
    / "projects/waferllm/batched_pair_rotation_5x1024_8x8_x/run-20260907T230157971748Z"
)
c = ROOT / "evidence/batched-pair-source-20260907T230412005447Z"
probe = ROOT / "evidence/dsd-base-offset-20260907T231325901420Z"
verify(c)
verify(probe)
assert read(probe / "offset-review.json")["passed"]
r, z = read(h / "results.json"), read(c / "results.json")
bs = read(h / "batches.json")
assert r["success"] and z["success"] and len(z["cases"]) == len(bs) == 6
rows = []
for e, (a, b, batch, out) in enumerate(
    zip(r["diagnostics"], z["diagnostics"], bs, z["cases"])
):
    differences = {
        k: int(np.count_nonzero(np.asarray(a[k]) != np.asarray(b[k])))
        for k in ("x", "cosine", "sine", "result", "history", "progress")
    }
    try:
        check(5, 1024, True, "odd_even", batch, out)
        numeric = True
    except AssertionError:
        numeric = False
    rows.append(
        dict(
            epoch=e,
            mismatched_words=differences,
            source_independent_math_passed=numeric,
        )
    )
assert all(
    not v["source_independent_math_passed"] for v in [rows[i] for i in (0, 1, 2, 5)]
)
files = [
    h / "manifest.json",
    h / "results.json",
    c / "provenance.json",
    c / "execution.json",
    c / "results.json",
    probe / "provenance.json",
    probe / "execution.json",
    probe / "results.json",
    probe / "offset-review.json",
    Path(__file__),
]
p = ROOT / "evidence/decode-pair-original-offset-failure.json"
assert not p.exists()
p.write_text(
    json.dumps(
        dict(
            passed=True,
            meaning="Failure reproduced, not source numerical qualification",
            checks=rows,
            cause="Base reset discards initial odd offset1, verified by separate SDK probe. Source repair must explicitly add1 after reset; source arithmetic convention remains explicit odd_even.",
            files={str(v.relative_to(ROOT)): sha(v) for v in files},
        ),
        indent=2,
    )
    + "\n"
)
print("ORIGINAL DECODE OFFSET FAILURE PRESERVED")
