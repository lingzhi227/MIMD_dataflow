"""Bind tighter RMS range metadata to unchanged CSL and full observed FFN rows."""

import argparse, hashlib, json, sys
from fractions import Fraction
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("sdk_bundle", type=Path)
p.add_argument("native_bundle", type=Path)
p.add_argument("report", type=Path)
a = p.parse_args()
old, new = a.sdk_bundle.resolve(), a.native_bundle.resolve()
assert not a.report.exists()
sys.path.insert(0, str(new / "implementation"))
from integrity import verify_bundle
from mesh_batched_feed_forward import plan

# Validate the old archive and its own snapshot, while executing the new analysis.
verify_bundle(old, implementation=False)
verify_bundle(new)
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
s = read(new / "schedule.json")
m = read(new / "semantic.json")
assert s == plan(m)
assert (
    read(old / "qualification.json")["success"]
    and read(old / "results.json")["success"]
)
assert (
    read(new / "application-gate.json")["passed"]
    and read(new / "target-application-gate.json")["passed"]
)
assert read(old / "batches.json") == read(new / "batches.json")
for f in new.glob("*.csl"):
    assert f.read_bytes() == (old / f.name).read_bytes(), f.name
l1 = s["numerical_bounds"]["normalized_l1"]
factor = Fraction(*map(int, l1["square_sum_lower_factor_exact"]))
loss = Fraction(2 * s["N"] + 1, 2 * 2**24)
rows = 0
max_norm = Fraction(0)
for batch, d in zip(
    read(old / "batches.json"), read(old / "results.json")["diagnostics"]
):
    x = np.asarray(batch["x"]).reshape(s["B"], s["N"])
    norm = (
        np.asarray(d["normalized"], np.uint16)
        .view(np.float16)[:, 0]
        .reshape(s["P"], s["B"], s["Nt"])
        .transpose(1, 0, 2)
        .reshape(s["B"], s["N"])
    )
    sums = np.asarray(d["sums"], np.uint16).view(np.float16)[0, 0]
    for i, row in enumerate(x):
        true_sum = sum(Fraction(float(v)) ** 2 for v in row)
        assert Fraction(float(sums[i])) >= factor * true_sum - loss
        actual = sum(Fraction(abs(float(v))) for v in norm[i])
        assert actual <= Fraction(l1["bound"])
        max_norm = max(max_norm, actual)
        rows += 1
    for port, bound in [
        ("partial", max(v["local"] for v in s["numerical_bounds"]["projections"])),
        (
            "projections",
            max(v["reduced"] for v in s["numerical_bounds"]["projections"]),
        ),
        ("hidden", s["numerical_bounds"]["hidden"]),
        ("delta", s["numerical_bounds"]["delta"]),
        ("result", s["numerical_bounds"]["result"]),
    ]:
        assert (
            np.max(
                np.abs(np.asarray(d[port], np.uint16).view(np.float16).astype(float))
            )
            <= bound
        ), port
report = dict(
    passed=True,
    actual_sdk_rows=rows,
    max_observed_normalized_l1=float(max_norm),
    new_bounds=s["numerical_bounds"],
    csl_identical=True,
    new_sdk_execution=False,
    scope="Exact rational lower-sum and L1 inequalities checked against all completed original-domain SDK rows; native sealed stages pass; tighter analysis does not change CSL. No expanded-input SDK qualification implied.",
    files={
        str(p): sha(p)
        for p in (
            old / "manifest.json",
            old / "results.json",
            old / "qualification.json",
            new / "manifest.json",
            new / "application-gate.json",
            new / "target-application-gate.json",
            Path(__file__),
        )
    },
)
a.report.write_text(json.dumps(report, indent=2) + "\n")
print("L1 PROOF/CSL/OBSERVATIONS PASS", rows)
