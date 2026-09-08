"""Ensure saved SDK state corruption is rejected without modifying evidence."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import json, shutil, struct, subprocess, sys, tempfile
from pathlib import Path

source = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "case"
    shutil.copytree(
        source,
        root,
        ignore=shutil.ignore_patterns(
            "out.core", "simfab_traces", "executables", "generated"
        ),
    )
    results = json.loads((root / "results.json").read_text())
    actors = [k for k, v in results["diagnostics"].items() if "state" in v]
    assert actors, "probe needs resident state"
    results["diagnostics"][actors[0]]["state"][0] = struct.unpack(
        "I", struct.pack("f", 999.0)
    )[0]
    (root / "results.json").write_text(json.dumps(results))
    result = subprocess.run(
        [sys.executable, str(root / "implementation/validate.py"), str(root)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "resident state" in result.stderr, (
        result.stdout + result.stderr
    )
    print(json.dumps({"state_corruption_rejected": True, "source": str(source)}))
