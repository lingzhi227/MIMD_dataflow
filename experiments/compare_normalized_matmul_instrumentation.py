"""Same HLS algorithm/batches/code policy, differing only observation mode."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, sys, subprocess, hashlib
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("sampled", type=Path)
p.add_argument("counters", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()


def audit(root):
    code = 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
    return json.loads(
        subprocess.check_output(
            [sys.executable, "-c", code, str(root.resolve())], text=True
        )
    )


reports = [audit(root) for root in (a.sampled, a.counters)]
assert all(r["passed"] for r in reports)
for name in ("layout.csl", "pe.csl", "inference_comm.csl", "inference_routes.csl"):
    assert (a.sampled / name).read_bytes() == (a.counters / name).read_bytes()


def read(root, name):
    return json.loads((root / name).read_text())


s, c = [read(root, "schedule.json") for root in (a.sampled, a.counters)]
assert s["instrumentation"] == "sampled" and c["instrumentation"] == "counters"
assert {
    k: v for k, v in s.items() if k not in ("instrumentation", "memory_per_pe")
} == {k: v for k, v in c.items() if k not in ("instrumentation", "memory_per_pe")}
for name in ("source.cpp", "batches.json", "runtime-options.json"):
    assert (a.sampled / name).read_bytes() == (a.counters / name).read_bytes()
rs, rc = [read(root, "results.json") for root in (a.sampled, a.counters)]
cases = []
for i, (ds, dc) in enumerate(zip(rs["diagnostics"], rc["diagnostics"])):
    for key in ("X", "W", "projection_input", "result", "progress"):
        np.testing.assert_array_equal(ds[key], dc[key])
    sc = reports[0]["cases"][i]["max_local_cycles"]
    cc = reports[1]["cases"][i]["max_local_cycles"]
    cases.append(
        dict(
            packed_half_bits_exact=True,
            sampled_max_local_cycles=sc,
            counter_max_local_cycles=cc,
            sampled_to_counter_ratio=sc / cc,
        )
    )
a.output.write_text(
    json.dumps(
        dict(
            passed=True,
            new_sdk_execution=False,
            scope="Same HLS numerical schedule and inputs; observation cost in local simulator interval.",
            cases=cases,
            hashes={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for root in (a.sampled, a.counters)
                for p in (root / "manifest.json", root / "results.json")
            },
        ),
        indent=2,
    )
    + "\n"
)
print(a.output)
