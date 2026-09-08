"""Sequential frozen SDK jobs, stopping on first failure and mirroring evidence."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import json, subprocess, sys
from pathlib import Path

ROOT = repository_root(__file__)
REMOTE = "/home/lingzhi/cerebras/csl-hls/ports"
for job in json.loads(Path(sys.argv[1]).read_text()):
    bundle = job["bundle"]
    assert bundle.startswith("benchmarks/") and all(
        c.isalnum() or c in "/_.-" for c in bundle
    )
    timeout = int(job["timeout"])
    assert timeout > 0
    print("START", bundle, flush=True)
    cmd = f"cd {REMOTE} && .venv/bin/python {bundle}/sdk-execution-driver.py {bundle} --timeout {timeout}"
    result = subprocess.run(["ssh", "workstation", cmd])
    subprocess.run(
        ["rsync", "-az", f"workstation:{REMOTE}/{bundle}/", str(ROOT / bundle) + "/"],
        check=True,
    )
    if result.returncode:
        raise SystemExit(result.returncode)
    print("MIRRORED", bundle, flush=True)
