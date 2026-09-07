"""Sequential frozen SDK jobs, stopping on first failure and mirroring evidence."""

import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/home/lingzhi/cerebras/csl-hls/ports"
for job in json.loads(Path(sys.argv[1]).read_text()):
    bundle = job["bundle"]
    assert bundle.startswith("projects/") and all(
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
