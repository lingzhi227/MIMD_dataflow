"""Apply the codegen preflight to an existing index without editing its snapshots."""

import argparse, datetime, hashlib, inspect, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from integrity import verify_codegen

p = argparse.ArgumentParser()
p.add_argument("index", type=Path)
a = p.parse_args()
index = json.loads(a.index.read_text())
rows = []
source = inspect.getsource(verify_codegen)
for case in index["cases"]:
    bundle = ROOT / case["artifact"]
    code = (
        'import sys,json;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from integrity import verify_bundle\n'
        + source
        + "\nprint(json.dumps(verify_codegen(p)))"
    )
    r = subprocess.run(
        [sys.executable, "-c", code, str(bundle.resolve())],
        capture_output=True,
        text=True,
    )
    rows.append(
        dict(
            artifact=case["artifact"],
            passed=r.returncode == 0,
            report=json.loads(r.stdout) if r.returncode == 0 else r.stderr,
            manifest_sha256=hashlib.sha256(
                (bundle / "manifest.json").read_bytes()
            ).hexdigest(),
        )
    )
    print(case["artifact"], r.returncode == 0, flush=True)
f = (
    ROOT
    / "evidence"
    / (
        "frozen-codegen-regression-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + ".json"
    )
)
f.write_text(
    json.dumps(
        dict(
            passed=all(x["passed"] for x in rows),
            cases=rows,
            new_sdk_execution=False,
            index_sha256=hashlib.sha256(a.index.read_bytes()).hexdigest(),
            preflight_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
            scope="Current dependency-completeness preflight applied using each immutable bundle backend and integrity checker. No numerical rerun or qualification upgrade.",
        ),
        indent=2,
    )
    + "\n"
)
print(f.relative_to(ROOT))
assert all(x["passed"] for x in rows)
