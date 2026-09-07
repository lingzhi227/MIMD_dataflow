"""Separate original half reduction, observed scale indexing and standard RMSNorm."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from probe_runtime import read, verify, sha

parser = argparse.ArgumentParser()
parser.add_argument("bundle", type=Path)
parser.add_argument("output", type=Path)
a = parser.parse_args()
assert not a.output.exists()
verify(a.bundle)
execution = read(a.bundle / "execution.json")
assert execution["success"] and execution["results_sha256"] == sha(
    a.bundle / "results.json"
)
results = read(a.bundle / "results.json")
inputs = read(a.bundle / "inputs.json")
assert results["success"] and len(results["cases"]) == len(inputs) == 3
p, t, n = 8, 8, 64
h = lambda v: np.asarray(v, dtype=np.float16)
bits = lambda v: np.asarray(v, np.float16).view(np.uint16)
rows = []
for epoch, (b, r) in enumerate(zip(inputs, results["cases"])):
    x = h(b["x"]).reshape(n, n)
    assert np.all(h(b["w"]) == 1)
    packed = h(x.reshape(p, t, p, t).transpose(0, 2, 3, 1).reshape(p, p, t * t))
    local = np.zeros((p, p, t), np.float16)
    columns = packed.reshape(p, p, t, t)
    for col in range(t):
        local = h(local + h(columns[:, :, col, :] * columns[:, :, col, :]))
    left = local[:, 0, :]
    for col in range(1, p // 2):
        left = h(left + local[:, col, :])
    right = local[:, -1, :]
    for col in range(p - 2, p // 2, -1):
        right = h(right + local[:, col, :])
    # mv_left_recv is routed from EAST: right chain arrives first.
    total = h(h(local[:, p // 2, :] + right) + left)
    reduced = np.asarray(r["hls_reduced"], np.uint16)
    np.testing.assert_array_equal(
        reduced, np.broadcast_to(bits(total)[:, None, :], (p, p, t))
    )
    inverse = np.asarray(r["hls_inverse"], np.uint16).view(np.float16)
    actual = np.asarray(r["hls_result"], np.uint16).view(np.float16).reshape(p, p, t, t)
    assert np.all(np.isfinite(inverse)) and np.all(np.isfinite(actual))
    np.testing.assert_array_equal(
        bits(inverse), np.broadcast_to(bits(inverse[:, 0, :])[:, None, :], (p, p, t))
    )
    # Source applies inverse[col] to the whole contiguous row-vector of that column.
    source_indexed = h(columns * inverse[:, :, :, None])
    np.testing.assert_array_equal(bits(actual), bits(source_indexed))
    nominal_sqrt = h(
        np.sqrt(h(h(total / np.float16(n)) + np.float16(1e-6)).astype(np.float64))
    )
    nominal_inverse = h(np.float16(1) / nominal_sqrt)
    inv_bits_exact = np.array_equal(bits(inverse[:, 0, :]), bits(nominal_inverse))
    gathered = actual.transpose(0, 3, 1, 2).reshape(n, n).astype(np.float64)
    xf = x.astype(np.float64)
    standard = xf / np.sqrt(np.mean(xf * xf, axis=1, keepdims=True) + 1e-6)
    denom = float(np.linalg.norm(standard))
    err = float(np.linalg.norm(gathered - standard))
    if denom == 0:
        np.testing.assert_array_equal(gathered, standard)
    np.testing.assert_array_equal(np.asarray(r["hls_progress"]), epoch + 1)
    rows.append(
        dict(
            epoch=epoch,
            independent_half_reduction_bits_exact=True,
            observed_feature_index_scaling_bits_exact=True,
            nominal_half_inverse_bits_exact=bool(inv_bits_exact),
            standard_rmsnorm_relative_l2=err / denom if denom else 0.0,
            standard_rmsnorm_max_abs_error=float(np.max(np.abs(gathered - standard))),
        )
    )
a.output.write_text(
    json.dumps(
        dict(
            observed_source_behavior_confirmed=True,
            standard_rmsnorm_qualification=False,
            new_sdk_execution=False,
            scope="Pinned original distributed RMS stage; arithmetic unchanged. Independent half tree sum and observed factor-index multiplication agree. Standard mathematical RMSNorm errors are reported separately; successful SDK execution does not certify the source naming.",
            bundle=str(a.bundle),
            results_sha256=sha(a.bundle / "results.json"),
            analyzer_sha256=sha(Path(__file__)),
            cases=rows,
        ),
        indent=2,
    )
    + "\n"
)
print(a.output)
print(rows)
