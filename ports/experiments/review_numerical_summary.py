"""Preserve checks/results; append an explicit correction to numerical summary labels."""

import argparse, copy, datetime, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_ports import numerical_summary

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
    / "evidence"
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
            summary_implementation_sha256=sha(ROOT / "run_ports.py"),
            correction="Remove inapplicable inherited f32 componentwise rtol/atol labels; state the normwise half limits actually checked, and softmax row mass/nonnegativity. Original independent checks, raw outputs, execution audits and artifacts preserved.",
            cases=cases,
        ),
        indent=2,
    )
    + "\n"
)
print(out.relative_to(ROOT), len(cases))
