"""Compare matched sampled/lean FFT runs after each frozen auditor passes."""

import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("sampled", type=Path)
p.add_argument("counters", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()


def read(root, name):
    return json.loads((root / name).read_text())


def audited(root):
    code = 'import sys,json;from pathlib import Path;p=Path(sys.argv[1]).resolve();sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
    return json.loads(
        subprocess.check_output([sys.executable, "-c", code, str(root)], text=True)
    )


sa, ca = audited(a.sampled), audited(a.counters)
assert (
    sa["passed"]
    and ca["passed"]
    and sa["internal_phases_observed"]
    and not ca["internal_phases_observed"]
)
for name in ("source.cpp", "batches.json", "runtime-options.json"):
    assert (a.sampled / name).read_bytes() == (a.counters / name).read_bytes(), name
s, c = [read(r, "schedule.json") for r in (a.sampled, a.counters)]
for key in (
    "profile",
    "rows",
    "cols",
    "N",
    "T",
    "local_length",
    "epochs",
    "transform",
    "input",
    "output",
    "nodes",
    "stages",
):
    assert s[key] == c[key], key
for key in ("source_sha256", "sdk_sha256"):
    assert (
        read(a.sampled, "qualification.json")[key]
        == read(a.counters, "qualification.json")[key]
    ), key
sr, cr = [read(r, "results.json") for r in (a.sampled, a.counters)]
rows = []
for epoch, (sd, cd) in enumerate(zip(sr["diagnostics"], cr["diagnostics"])):
    np.testing.assert_array_equal(
        np.asarray(sd["packed_output"], np.float32).view(np.uint32),
        np.asarray(cd["packed_output"], np.float32).view(np.uint32),
    )
    np.testing.assert_array_equal(sd["progress"], cd["progress"])
    sc = sa["cases"][epoch]["max_local_cycles"]
    cc = ca["cases"][epoch]["max_local_cycles"]
    rows.append(
        dict(
            epoch=epoch,
            packed_device_bits_exact=True,
            sampled_max_local_cycles=sc,
            counters_max_local_cycles=cc,
            sampled_over_counters=sc / cc,
        )
    )
a.output.write_text(
    json.dumps(
        dict(
            passed=True,
            new_sdk_execution=False,
            sources=[
                dict(
                    bundle=str(root),
                    results_sha256=hashlib.sha256(
                        (root / "results.json").read_bytes()
                    ).hexdigest(),
                )
                for root in (a.sampled, a.counters)
            ],
            comparisons=rows,
            scope="Same HLS algorithm/input/region/direction/norm/SDK options. Sampled mode observes seven pencil-endpoint stages; counters mode has none. Maximum-local simulator intervals, not global/hardware latency.",
        ),
        indent=2,
    )
    + "\n"
)
print(a.output)
