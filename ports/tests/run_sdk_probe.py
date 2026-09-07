"""SDK regression probe for bool lowering, scalar wire padding, and persistent state."""

import datetime, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from compile import build
from validate import audit

out = (
    ROOT
    / "evidence"
    / (
        "probe-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
)
build(
    ROOT / "tests/probes/boolean.cpp",
    out,
    epochs=3,
    bound=4,
    batches=[{"a": [2.0]}, {"a": [-2.0]}, {"a": [3.0]}],
)
env = dict(
    os.environ, SINGULARITYENV_CS_TARGET="SDR", SINGULARITYENV_PYTHONUNBUFFERED="1"
)
with (out / "sdk.log").open("w") as log:
    subprocess.run(
        [
            "/home/lingzhi/cerebras/sdk/2.10.1/cs_python",
            str(out / "implementation/sdk.py"),
            str(out),
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        timeout=240,
        check=True,
    )
assert json.loads((out / "results.json").read_text())["cases"] == [
    {"result": [2.0]},
    {"result": [2.0]},
    {"result": [5.0]},
]
report = audit(out)
(out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(out, report)
