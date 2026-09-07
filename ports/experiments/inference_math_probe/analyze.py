"""Quantify SDK functions and source polynomial separately after real execution."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("bundle", type=Path)
parser.add_argument("output", type=Path)
a = parser.parse_args()
assert not a.output.exists()
read = lambda name: json.loads((a.bundle / name).read_text())
for name, digest in read("provenance.json")["files"].items():
    assert hashlib.sha256((a.bundle / name).read_bytes()).hexdigest() == digest, name
assert read("execution.json")["success"] and read("results.json")["success"]
inputs = read("inputs.json")
results = read("results.json")["cases"]
assert len(inputs) == len(results) == 3
rows = []
half = lambda x: float(np.float16(x))
for epoch, (values, result) in enumerate(zip(inputs, results)):
    x = np.asarray(values, np.float16)
    np.testing.assert_array_equal(x.view(np.uint16), result["x"])
    assert result["progress"] == [epoch + 1]
    source = []
    for v in x:
        t = half(1 + half(float(v) / 256))
        t = half(t * t)
        source.append(half(t * t))
    np.testing.assert_array_equal(
        np.asarray(source, np.float16).view(np.uint16), result["source_exp"]
    )
    observed = {
        k: np.asarray(result[k], np.uint16).view(np.float16).astype(np.float64)
        for k in ("sdk_exp", "source_exp", "sdk_sqrt")
    }
    assert all(np.all(np.isfinite(v)) for v in observed.values())
    refs = dict(
        sdk_exp=np.exp(x.astype(np.float64)),
        sdk_sqrt=np.sqrt(np.abs(x.astype(np.float64))),
    )
    metrics = {}
    for name, ref in refs.items():
        nearest = ref.astype(np.float16)
        # Zero sign is reported in raw bits, not treated as an accuracy failure.
        actual_magnitude = np.abs(observed[name]).astype(np.float16)
        metrics[name] = dict(
            max_abs_error=float(np.max(np.abs(observed[name] - ref))),
            max_nearest_half_ulp_distance=int(
                np.max(
                    np.abs(
                        actual_magnitude.view(np.uint16).astype(np.int32)
                        - nearest.view(np.uint16).astype(np.int32)
                    )
                )
            ),
        )
    rows.append(
        dict(
            epoch=epoch,
            source_polynomial_bits_exact=True,
            metrics=metrics,
            source_exp_max_abs_difference_from_standard_exp=float(
                np.max(np.abs(observed["source_exp"] - refs["sdk_exp"]))
            ),
        )
    )
for name in ("sdk_exp", "source_exp", "sdk_sqrt"):
    assert results[1][name] == results[0][name][::-1], (
        "Changed input repeat consistency: " + name
    )
out = dict(
    passed=True,
    new_sdk_execution=False,
    scope="Three real changed-input calls; typed transport, completion and source polynomial bit checks. SDK exp/sqrt accuracy is quantified on these inputs, not a universal precision guarantee or an inference qualification.",
    bundle=str(a.bundle),
    analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    numpy_version=np.__version__,
    hashes={
        name: hashlib.sha256((a.bundle / name).read_bytes()).hexdigest()
        for name in ("results.json", "inputs.json", "execution.json", "provenance.json")
    },
    cases=rows,
    raw_output_bits=results,
)
a.output.write_text(json.dumps(out, indent=2) + "\n")
print(a.output)
