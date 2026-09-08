"""Independent exact-zero check for unexecuted factorization checkpoints."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse
import hashlib
import json
from pathlib import Path


def audit(root):
    root = Path(root)
    s = json.loads((root / "schedule.json").read_text())
    r = json.loads((root / "results.json").read_text())
    if s["profile"] not in ("mesh_lu.v1", "mesh_cholesky.v1") or not r["success"]:
        raise ValueError("expected completed factorization")
    count = 0
    for diag in r["diagnostics"]:
        for y in range(s["P"]):
            for x in range(s["P"]):
                limit = (
                    (min(x, y) + 1) * s["Nt"]
                    if s["profile"] == "mesh_lu.v1"
                    else ((x + 1) * s["Nt"] if x <= y else 0)
                )
                corners = diag["checkpoints"][y][x]
                if len(corners) != s["N"] or any(len(v) != 2 for v in corners):
                    raise ValueError("checkpoint dimensions")
                for pair in corners[limit:]:
                    if pair != [0.0, 0.0]:
                        raise ValueError(f"nonzero unexecuted checkpoint p{x}_{y}")
                    count += 2
    return {
        "bundle": str(root),
        "passed": True,
        "exact_zero_entries": count,
        "results_sha256": hashlib.sha256(
            (root / "results.json").read_bytes()
        ).hexdigest(),
        "schedule_sha256": hashlib.sha256(
            (root / "schedule.json").read_bytes()
        ).hexdigest(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundles", nargs="+")
    p.add_argument("-o", required=True)
    a = p.parse_args()
    report = {
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [audit(root) for root in a.bundles],
    }
    Path(a.o).write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(report['cases'])} exact checkpoint audits passed")
