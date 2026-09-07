"""Same HLS algorithm/batches/code policy, differing only observation mode."""

import argparse, json, sys, importlib
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("sampled", type=Path)
p.add_argument("counters", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
sys.path.insert(0, str(a.sampled / "implementation"))
from validate import audit

assert (
    json.loads((a.sampled / "manifest.json").read_text())["implementation"]
    == json.loads((a.counters / "manifest.json").read_text())["implementation"]
), "same frozen auditor implementation required"
reports = [audit(root) for root in (a.sampled, a.counters)]


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
    np.testing.assert_array_equal(ds["result"], dc["result"])
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
        ),
        indent=2,
    )
    + "\n"
)
print(a.output)
