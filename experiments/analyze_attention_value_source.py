"""Separate source-exact behavior from logical probability-times-V correctness."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, math, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from mesh_common import pack_tiles, unpack_tiles
from mesh_twohop import cycle, block_index

p = argparse.ArgumentParser()
p.add_argument("probe", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
verify(a.probe)
e = read(a.probe / "execution.json")
assert e["success"] and e["results_sha256"] == sha(a.probe / "results.json")
r = read(a.probe / "results.json")
bs = read(a.probe / "inputs.json")
logical = read(a.probe / "logical-inputs.json")
g = read(a.probe / "geometry.json")
m, n, P = g["M"], g["N"], g["P"]
mt, nt = m // P, n // P
order = cycle(P)
bits = lambda v: np.asarray(v, np.float16).view(np.uint16)
device_layout = read(a.probe / "provenance.json").get("device_layout_adapter", False)
assert (
    r["success"]
    and r["runtime_instances"] == 1
    and len(r["cases"]) == len(bs) == len(logical) == (2 if device_layout else 4)
)
sampled = (
    read(a.probe / "provenance.json").get("instrumentation", "sampled") == "sampled"
)
reports = []
for epoch, (b, original, d) in enumerate(zip(bs, logical, r["cases"])):
    probability = np.asarray(original["probability"]).reshape(m, m)
    value = np.asarray(original["value"]).reshape(m, n)
    q = pack_tiles(probability, P, P, "F")
    v = np.asarray(b["value"])
    out = np.zeros((P, P, mt * nt))
    prefix = np.zeros((P, P, P, mt * nt))
    for y in range(P):
        for x in range(P):
            acc = np.zeros((mt, nt))
            for step in range(P):
                left = q[y, block_index(P, y, x, step)].reshape(mt, mt, order="F")
                owner = order[(order.index(y) - step) % P]
                if device_layout:
                    owner = block_index(P, y, x, step)
                right = v[owner, x].reshape(mt, nt, order="F" if device_layout else "C")
                for k in range(mt):
                    acc = np.asarray(
                        acc + left[:, k, None] * right[None, k, :], np.float16
                    ).astype(float)
                prefix[y, x, step] = acc.ravel(order="F")
            out[y, x] = acc.ravel(order="F")
    raw = np.asarray(d["output"], np.uint16)
    actual = unpack_tiles(raw.view(np.float16), mt, nt, "F").astype(float)
    mismatches = int(np.count_nonzero(raw != bits(out)))
    history_mismatches = None
    if sampled:
        history_mismatches = int(
            np.count_nonzero(
                np.asarray(d["history"], np.uint16) != bits(prefix.reshape(P, P, -1))
            )
        )
    else:
        np.testing.assert_array_equal(d["history"], np.zeros((P, P, 1), np.uint16))
    standard = np.array(
        [
            [
                math.fsum(float(x) * float(y) for x, y in zip(row, value[:, col]))
                for col in range(n)
            ]
            for row in probability
        ]
    )
    err = actual - standard
    l2 = float(np.linalg.norm(err)) / max(float(np.linalg.norm(standard)), 1e-30)
    peak = float(np.max(np.abs(err))) / max(float(np.max(np.abs(standard))), 1e-30)
    np.testing.assert_array_equal(d["progress"], epoch + 1)
    reports.append(
        dict(
            layout=original["layout"],
            source_behavior_half_mismatches=mismatches,
            prefix_half_mismatches=history_mismatches,
            prefix_observed=sampled,
            standard_relative_l2=l2,
            standard_peak_scaled_error=peak,
            mathematical_accuracy_passed=l2 <= 0.015 and peak <= 0.02,
        )
    )
report = dict(
    source_behavior_confirmed=all(
        x["source_behavior_half_mismatches"] == 0
        and x["prefix_half_mismatches"] in (0, None)
        for x in reports
    ),
    logical_input_layout_compatible=all(
        x["mathematical_accuracy_passed"] for x in reports[:2]
    ),
    prealigned_row_major_diagnostic_passed=(
        None
        if device_layout
        else all(x["mathematical_accuracy_passed"] for x in reports[2:])
    ),
    device_layout_adapter=device_layout,
    cases=reports,
    new_sdk_execution=False,
    scope=(
        "Isolated probability-times-V source with device vertical prealignment and strided DSD reads. Logical column-major inputs; no host prealignment or local transpose copy. Source-adapter experiment, not HLS qualification."
        if device_layout
        else "Isolated original probability-times-V source. Host-prealigned row-major input is a diagnostic, not resident device conversion or HLS qualification."
    ),
    hashes={
        str(x): sha(x)
        for x in [
            a.probe / "provenance.json",
            a.probe / "results.json",
            Path(__file__),
            ROOT / "lib/Conversion/mesh_twohop.py",
            ROOT / "lib/Conversion/mesh_common.py",
        ]
    },
)
a.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
