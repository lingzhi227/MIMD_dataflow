"""Preserve the Q32 failure and verify Q4 changed scheduling, not inputs or gates."""

import argparse, difflib, hashlib, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("failed", type=Path)
p.add_argument("passing", type=Path)
p.add_argument("report", type=Path)
a = p.parse_args()
assert not a.report.exists()
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
fm, pm = [read(b / "manifest.json") for b in (a.failed, a.passing)]
mismatches = {
    n: dict(recorded=h, actual=sha(a.failed / n))
    for n, h in fm["files"].items()
    if sha(a.failed / n) != h
}
assert set(mismatches) == {"stage.json"}, mismatches
assert all(sha(a.passing / n) == h for n, h in pm["files"].items())
for name in ("batches.json", "application-reference.py"):
    assert (a.failed / name).read_bytes() == (a.passing / name).read_bytes(), name
old, new = [(b / "source.cpp").read_text() for b in (a.failed, a.passing)]
# Exactly one block-size scheduling directive changes; arithmetic and source
# numerical oracle remain fixed. No failed bundle files are repaired here.
assert old.count("(normalized,wq,32)") == 1
assert old.replace("(normalized,wq,32)", "(normalized,wq,4)") == new
fg = read(a.failed / "application-gate.json")
pg = read(a.passing / "application-gate.json")
tg = read(a.passing / "target-application-gate.json")
assert not fg["passed"] and pg["passed"] and tg["passed"]
assert len(pg["checks"]) == len(tg["checks"]) == 8
report = dict(
    passed=True,
    failed_bundle=str(a.failed),
    passing_bundle=str(a.passing),
    failed_gate=fg,
    passing_native_checks=pg["checks"],
    passing_target_checks=tg["checks"],
    legacy_failed_stage_hash_mismatch=mismatches,
    verified_failed_entries=len(fm["files"]) - len(mismatches),
    verified_passing_entries=len(pm["files"]),
    source_diff=list(difflib.unified_diff(old.splitlines(), new.splitlines())),
    scope="Preserved native-only failure. Original failure gate updated stage.json after manifest sealing; exactly that historical mismatch is recorded, never repaired. Q4 is a scheduling correction under identical inputs and fixed numerical oracle. SDK execution is separate.",
    files={
        str(b / n): sha(b / n)
        for b in (a.failed, a.passing)
        for n in (
            "manifest.json",
            "application-gate.json",
            "stage.json",
            "source.cpp",
            "batches.json",
            "application-reference.py",
        )
    },
    driver_sha256=sha(Path(__file__)),
)
a.report.write_text(json.dumps(report, indent=2) + "\n")
print("ITERATION REVIEW PASS")
