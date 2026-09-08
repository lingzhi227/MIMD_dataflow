"""Supplement old QR bundles with input-independent schedule/count/role checks."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from qr_schedule import rotations
from mesh_qr_sdk import audit_rotations


def audit(root):
    root = Path(root)
    s = json.loads((root / "schedule.json").read_text())
    r = json.loads((root / "results.json").read_text())
    if s["profile"] != "mesh_qr.v1" or not r["success"]:
        raise ValueError("completed QR required")
    observations = []
    for epoch, d in enumerate(r["diagnostics"]):
        for y in range(s["rows"]):
            for x in range(s["cols"]):
                expected = rotations(s["rows"], s["cols"], s["Nt"], x, y)
                check = audit_rotations(
                    d["witnesses"][y][x], d["progress"][y][x][0], expected
                )
                observations.append(
                    dict(
                        epoch=epoch,
                        node=f"p{x}_{y}",
                        expected_count=len(expected),
                        **check,
                    )
                )
    return {
        "passed": True,
        "bundle": str(root),
        "observations": observations,
        "results_sha256": hashlib.sha256(
            (root / "results.json").read_bytes()
        ).hexdigest(),
        "implementation_sha256": {
            name: hashlib.sha256((ROOT / "lib" / name).read_bytes()).hexdigest()
            for name in ("qr_schedule.py", "mesh_qr_sdk.py")
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle")
    p.add_argument("-o", required=True)
    a = p.parse_args()
    result = audit(a.bundle)
    Path(a.o).write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(result['observations'])} PE/epoch symbolic schedule checks passed")
