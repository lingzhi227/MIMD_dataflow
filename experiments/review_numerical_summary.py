"""Preserve checks/results; append an explicit correction to numerical summary labels."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, copy, datetime, hashlib, json, sys
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT))
from run_profiles import numerical_summary

p = argparse.ArgumentParser()
p.add_argument("report", type=Path)
a = p.parse_args()
r = json.loads(a.report.read_text())
assert r["success"]
cases = []
for c in r["cases"]:
    if (c.get("numerical_validation") or {}).get("contract") not in (
        "softmax-half-normwise-v1",
        "normalized-matmul-half-normwise-v1",
    ):
        continue
    new = copy.deepcopy(c)
    new["numerical_validation"] = numerical_summary(c)
    cases.append(new)
assert cases
prefix = "qualification-" if r.get("sdk") else "run-"
out = (
    ROOT
    / "validation/evidence"
    / (
        prefix
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + ".json"
    )
)
assert not out.exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
out.write_text(
    json.dumps(
        dict(
            kind="numerical_summary_metadata_correction",
            sdk=r.get("sdk", False),
            success=True,
            new_sdk_execution=False,
            new_native_execution=False,
            source_report=str(a.report),
            source_report_sha256=sha(a.report),
            summary_implementation_sha256=sha(ROOT / "tools/run_profiles.py"),
            correction="Remove inapplicable inherited f32 componentwise rtol/atol labels; state the normwise half limits actually checked, and softmax row mass/nonnegativity. Original independent checks, raw outputs, execution audits and artifacts preserved.",
            cases=cases,
        ),
        indent=2,
    )
    + "\n"
)
print(out.relative_to(ROOT), len(cases))
