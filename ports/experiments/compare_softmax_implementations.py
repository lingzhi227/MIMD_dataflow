"""Compare scalar/map lowering with frozen auditors and identical numerical inputs."""

import argparse, copy, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("scalar", type=Path)
p.add_argument("mapped", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()


def read(root, name):
    return json.loads((root / name).read_text())


def audit(root):
    code = 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
    return json.loads(
        subprocess.check_output(
            [sys.executable, "-c", code, str(root.resolve())], text=True
        )
    )


s, c = [read(root, "schedule.json") for root in (a.scalar, a.mapped)]
assert s.get("elementwise", "scalar") == "scalar" and c["elementwise"] == "map"
normalized = copy.deepcopy(c)
normalized.pop("elementwise")
normalized["stages"][2].pop("elementwise")
normalized["resources"].pop("elementwise_descriptors")
assert normalized == s
for name in ("batches.json", "runtime-options.json"):
    assert read(a.scalar, name) == read(a.mapped, name)
assert (a.scalar / "source.cpp").read_text() == (
    a.mapped / "source.cpp"
).read_text().replace(" elementwise=map", "")
reports = [audit(root) for root in (a.scalar, a.mapped)]
assert all(v["passed"] for v in reports)
rs, rm = [read(root, "results.json") for root in (a.scalar, a.mapped)]
assert len(rs["cases"]) == len(rm["cases"])
cases = []
for i, (ds, dm) in enumerate(zip(rs["diagnostics"], rm["diagnostics"])):
    for key in ("X", "result", "history", "exponents", "progress"):
        if ds[key] is None:
            assert dm[key] is None
        else:
            np.testing.assert_array_equal(ds[key], dm[key])
    sc = reports[0]["cases"][i]["max_local_cycles"]
    mc = reports[1]["cases"][i]["max_local_cycles"]
    cases.append(
        dict(
            all_observed_half_bits_exact=True,
            scalar_max_local_cycles=sc,
            map_max_local_cycles=mc,
            scalar_over_map=sc / mc,
            interval_reduction=1 - mc / sc,
        )
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


a.output.write_text(
    json.dumps(
        dict(
            passed=True,
            new_sdk_execution=False,
            scope="Same HLS algorithm/inputs/ownership/communication/instrumentation; scalar versus@map exponent implementation. WSE3 local simulator intervals, not hardware throughput.",
            cases=cases,
            hashes={
                str(p): sha(p)
                for root in (a.scalar, a.mapped)
                for p in (root / "manifest.json", root / "results.json")
            },
        ),
        indent=2,
    )
    + "\n"
)
print(a.output)
print(cases)
