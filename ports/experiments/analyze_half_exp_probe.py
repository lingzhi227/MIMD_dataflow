"""Qualify conversion/range reduction and every nonpositive finite half exp input."""

import argparse, json, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, verify, sha
from half_exp_candidate import model
from binary16 import bits

p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
verify(a.bundle)
e = read(a.bundle / "execution.json")
r = read(a.bundle / "results.json")
assert (
    e["success"]
    and e["results_sha256"] == sha(a.bundle / "results.json")
    and r["success"]
    and len(r["cases"]) == 2
)
summaries = []
for epoch, c in enumerate(r["cases"]):
    expected = np.arange(0x7C00, dtype=np.uint16) | 0x8000
    if epoch:
        expected = expected[::-1]
    np.testing.assert_array_equal(np.asarray(c["input"], np.uint16).ravel(), expected)
    np.testing.assert_array_equal(c["progress"], epoch + 1)
    x = expected.view(np.float16).astype(float)
    values = [model(v) for v in x]
    out = np.asarray(c["value"], np.uint16).ravel()
    rem = np.asarray(c["remainder"], np.uint16).ravel()
    n = np.asarray(c["exponent"], np.uint16).view(np.int16).ravel()
    np.testing.assert_array_equal(
        out, np.asarray([bits(v[0]) for v in values], np.uint16)
    )
    np.testing.assert_array_equal(
        rem, np.asarray([bits(v[2]) for v in values], np.uint16)
    )
    np.testing.assert_array_equal(n, [v[1] for v in values])
    nearest = np.exp(x).astype(np.float16).view(np.uint16)
    distance = np.abs(out.astype(int) - nearest.astype(int))
    summaries.append(
        dict(
            exact_sdk_model_bits=True,
            exact_nearest_integer_range_reduction=True,
            exact_half_remainder=True,
            non_nearest_half_exp_count=int(np.count_nonzero(distance)),
            max_nearest_half_ulp_distance=int(distance.max()),
        )
    )
a.output.write_text(
    json.dumps(
        dict(
            passed=True,
            new_sdk_execution=False,
            domain="All31744nonpositive finite half encodings including negativezero;8PEs,2changed warm calls. Positive inputs, other formats/flags not covered.",
            cases=summaries,
            hashes={
                str(p): sha(p)
                for p in [
                    a.bundle / "results.json",
                    a.bundle / "provenance.json",
                    Path(__file__),
                    Path(__file__).with_name("half_exp_candidate.py"),
                ]
            },
        ),
        indent=2,
    )
    + "\n"
)
print(a.output)
print(summaries)
