"""Check new optional map lowering preserves previously generated scalar schedules/CSL."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, sys, tempfile, hashlib
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from mesh_softmax import plan, generate

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
p.add_argument("bundles", type=Path, nargs="+")
a = p.parse_args()
assert not a.output.exists()
cases = []
for root in a.bundles:
    m = json.loads((root / "semantic.json").read_text())
    s = json.loads((root / "schedule.json").read_text())
    assert plan(m) == s
    with tempfile.TemporaryDirectory() as td:
        generate(s, td)
        for name in ("pe.csl", "layout.csl", "row_chain.csl"):
            assert (root / name).read_bytes() == (Path(td) / name).read_bytes()
    cases.append(
        dict(bundle=str(root), schedule_exact=True, generated_csl_bytes_exact=True)
    )
a.output.write_text(
    json.dumps(
        dict(
            passed=True,
            new_sdk_execution=False,
            scope="Optional map attribute leaves old scalar schedules and generated CSL bytes unchanged; this is a read-only compatibility check, not a new device run.",
            cases=cases,
            implementation_sha256=hashlib.sha256(
                (ROOT / "lib/Conversion/mesh_softmax.py").read_bytes()
            ).hexdigest(),
        ),
        indent=2,
    )
    + "\n"
)
print(a.output)
