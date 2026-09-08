"""Actual SDK-host C++ rebuild of a frozen batched RMS bundle."""

import json, hashlib, sys, shutil
from pathlib import Path

bundle, root = map(lambda p: Path(p).resolve(), sys.argv[1:])
assert not root.exists()
root.mkdir()
sys.path[:0] = [str(bundle / "implementation")]
from integrity import verify_bundle
from compile import build
from native_transport import parse_outputs

verify_bundle(bundle)
import importlib.util

spec = importlib.util.spec_from_file_location(
    "frozen_application", bundle / "application-reference.py"
)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
s = json.loads((bundle / "schedule.json").read_text())
bs = json.loads((bundle / "batches.json").read_text())
p = build(
    bundle / "source.cpp",
    root / "build",
    epochs=8,
    bound=2,
    batches=bs,
    instrumentation=s["instrumentation"],
    sdk_options=json.loads((bundle / "runtime-options.json").read_text()),
)
for f in p.glob("*.csl"):
    assert f.read_bytes() == (bundle / f.name).read_bytes(), f.name
checks = [
    app.check(s["B"], s["N"], s["F"], s["projections"], b, o)
    for b, o in zip(bs, parse_outputs((p / "native-output.txt").read_text()))
]
assert len(checks) == 8
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copyfile(__file__, root / "driver.py")
(root / "review.json").write_text(
    json.dumps(
        dict(
            passed=True,
            checks=checks,
            csl_identical=True,
            prior_manifest_sha256=sha(bundle / "manifest.json"),
        ),
        indent=2,
    )
    + "\n"
)
(root / "provenance.json").write_text(
    json.dumps(
        dict(
            files={
                str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()
            }
        ),
        indent=2,
    )
    + "\n"
)
print("PASS", root)
