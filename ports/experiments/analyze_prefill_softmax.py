"""Separate successful CSL execution from original softmax numerical correctness."""

import argparse, json
from pathlib import Path
import numpy as np
from probe_runtime import read, verify, sha

p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
verify(a.bundle)
e = read(a.bundle / "execution.json")
r = read(a.bundle / "results.json")
b = read(a.bundle / "inputs.json")
assert (
    e["success"]
    and e["results_sha256"] == sha(a.bundle / "results.json")
    and r["success"]
    and len(r["cases"]) == len(b) == 3
)
reports = []
for epoch, (case, values) in enumerate(zip(r["cases"], b)):
    x = np.asarray(values, float).reshape(64, 64)
    scaled = x * 0.125
    shifted = scaled - np.max(scaled, axis=1)[:, None]
    nominal = np.exp(shifted)
    nominal /= np.sum(nominal, axis=1)[:, None]
    raw = np.asarray(case["hls_score"], np.uint16)
    actual = (
        raw.view(np.float16)
        .reshape(8, 8, 8, 8)
        .transpose(0, 3, 1, 2)
        .reshape(64, 64)
        .astype(float)
    )
    np.testing.assert_array_equal(case["hls_progress"], epoch + 1)
    finite = bool(np.all(np.isfinite(actual)))
    if finite:
        relative = float(np.linalg.norm(actual - nominal) / np.linalg.norm(nominal))
        maximum = float(np.max(np.abs(actual - nominal)))
    else:
        relative = maximum = None
    reports.append(
        dict(
            epoch=epoch,
            finite=finite,
            nonfinite_outputs=int(np.count_nonzero(~np.isfinite(actual))),
            standard_relative_l2=relative,
            standard_max_abs=maximum,
            standard_accuracy_passed=finite and relative <= 0.01,
        )
    )
    if epoch == 1:
        assert np.all(x == -1024)
        assert np.all(np.asarray(case["hls_max"], np.uint16) == 0)
        assert np.all(np.asarray(case["hls_exp"], np.uint16) == 0)
        assert np.all(np.asarray(case["hls_sum"], np.uint16) == 0)
        assert np.all(np.asarray(case["hls_inverse"], np.uint16) == 0x7C00)
        assert np.all((raw & 0x7C00) == 0x7C00) and np.all((raw & 0x3FF) != 0)
    if epoch == 2:
        np.testing.assert_array_equal(actual, nominal)
assert (
    reports[0]["standard_accuracy_passed"]
    and not reports[1]["standard_accuracy_passed"]
    and reports[2]["standard_accuracy_passed"]
)
out = dict(
    observed_source_behavior_confirmed=True,
    standard_softmax_qualification=False,
    new_sdk_execution=False,
    cases=reports,
    scope="Pinned Prefill softmax score stage, alpha=1/sqrt64. Zero initialization of max leaves allnegative scaled scores at-128; SDKexp underflows all terms, sum0, reciprocalInf, then NaN outputs. Zero-input next warmcall recovers to exact uniform1/64. No arithmetic repair in this probe.",
    hashes={
        str(p): sha(p)
        for p in [
            a.bundle / "results.json",
            a.bundle / "provenance.json",
            Path(__file__),
        ]
    },
)
a.output.write_text(json.dumps(out, indent=2) + "\n")
print(a.output)
print(reports)
