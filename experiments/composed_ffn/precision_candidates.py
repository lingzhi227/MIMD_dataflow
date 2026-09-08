"""Native-IR sensitivity at the preserved cancellation failure; no SDK claim."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, json, hashlib, sys
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from projected_cache_ffn_reference import native_graph, original_math

r = ROOT / "validation/evidence/hls-composed-run-20260908T022401622593Z"
s = json.loads((r / "schedule.json").read_text())
row = json.loads((r / "batches.json").read_text())[6]
ideal = original_math(s, row)
reports = []
import numpy as np

for qblock, sblock in ((4, 16), (2, 16), (4, 8), (2, 8), (1, 8)):
    m = copy.deepcopy(s["graph"]["module"])
    by = {n["id"]: n for n in m["nodes"]}
    by["query"]["block_size"] = qblock
    by["score"]["block_size"] = sblock
    v = native_graph(m, row)
    error = v["score"] - ideal["score"]
    reports.append(
        dict(
            query_block=qblock,
            score_block=sblock,
            l2=float(np.linalg.norm(error) / np.linalg.norm(ideal["score"])),
            peak=float(np.max(np.abs(error)) / np.max(np.abs(ideal["score"]))),
        )
    )
p = ROOT / "validation/evidence/composed-precision-candidates.json"
assert not p.exists()
p.write_text(
    json.dumps(
        dict(
            scope="Native IR predicted sensitivity at fixed original epoch6. Candidate selection only; requires fresh C++ and actual CSL validation.",
            cases=reports,
            files={
                str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in [
                    r / "schedule.json",
                    r / "batches.json",
                    r / "native-math.json",
                    Path(__file__),
                    ROOT / "lib/Numerics/projected_cache_ffn_reference.py",
                ]
            },
        ),
        indent=2,
    )
    + "\n"
)
print(reports)
