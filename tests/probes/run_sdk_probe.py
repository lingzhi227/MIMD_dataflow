"""SDK regression probe for bool lowering, scalar wire padding, and persistent state."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import datetime, json, os, subprocess, sys
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from compile import build
from validate import audit

out = (
    ROOT
    / "validation/evidence"
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
