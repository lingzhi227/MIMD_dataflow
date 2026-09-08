"""Audit all finite positive half sqrt encodings and distinguish expression/FMA."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, math, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, verify, sha

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from binary16 import bits, quantize, fma
from sdk_math_reference import sqrt_f16, rms_inverse_f16

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
    and r["success"]
    and e["results_sha256"] == sha(a.bundle / "results.json")
)
assert len(r["cases"]) == 62
raw = {
    name: np.asarray([c[name] for c in r["cases"]], np.uint16).ravel()
    for name in read(a.bundle / "schema.json")["outputs"] + ["input"]
}
np.testing.assert_array_equal(raw["input"], np.arange(0x7C00, dtype=np.uint16))
x = raw["input"].view(np.float16).astype(np.float64)
model_sqrt = np.asarray([bits(sqrt_f16(v)) for v in x], np.uint16)
np.testing.assert_array_equal(raw["sqrt_value"], model_sqrt)
roots = model_sqrt.view(np.float16).astype(np.float64)
reciprocal = np.asarray(
    [bits(quantize(1 / v)) if v else 0x7C00 for v in roots], np.uint16
)
np.testing.assert_array_equal(raw["reciprocal"], reciprocal)
mean = np.asarray(x / 64, np.float16)
argument = np.asarray(mean + np.float16(1e-6), np.float16)
np.testing.assert_array_equal(raw["mean"], mean.view(np.uint16))
np.testing.assert_array_equal(raw["argument"], argument.view(np.uint16))
np.testing.assert_array_equal(
    raw["chain_sqrt"],
    np.asarray([bits(sqrt_f16(float(v))) for v in argument], np.uint16),
)
chain = np.asarray([bits(rms_inverse_f16(v, 64)) for v in x], np.uint16)
np.testing.assert_array_equal(raw["chain_inverse"], chain)
np.testing.assert_array_equal(raw["expression_chain"], chain)
norm = (np.arange(0x7C00, dtype=np.uint16) & 0x03FF) | 0x3C00
normal = norm.view(np.float16).astype(np.float64)
split = np.asarray([bits(quantize(quantize(v * v) - v)) for v in normal], np.uint16)
fused = np.asarray([bits(fma(v, v, -v)) for v in normal], np.uint16)
np.testing.assert_array_equal(raw["expr_fma"], split)
np.testing.assert_array_equal(raw["direct_fma"], fused)
nearest = np.sqrt(x).astype(np.float16).view(np.uint16)
distance = np.abs(model_sqrt.astype(np.int32) - nearest.astype(np.int32))
for epoch, c in enumerate(r["cases"]):
    assert c["progress"] == [epoch + 1]
out = dict(
    passed=True,
    new_sdk_execution=False,
    bundle=str(a.bundle),
    domain="All31744nonnegative finite IEEE-half input encodings,62warm calls,SDK2.10.1 default cslc flags. Signed negative zero/special values not part of this exhaustive set.",
    source_sqrt_model_bits_exact=True,
    language_reciprocal_nearest_half_bits_exact=True,
    staged_and_expression_rms_inverse_bits_exact=True,
    ordinary_expression_split_bits_exact=True,
    explicit_dsd_fma_bits_exact=True,
    split_vs_fused_different_values=int(np.count_nonzero(split != fused)),
    sdk_sqrt_max_nearest_half_ulp_distance=int(distance.max()),
    sdk_sqrt_non_nearest_count=int(np.count_nonzero(distance)),
    library_inv_vs_nearest_different_values=int(
        np.count_nonzero(raw["library_inv"] != reciprocal)
    ),
    library_inv_max_nearest_half_ulp_distance=int(
        np.max(
            np.abs(raw["library_inv"].astype(np.int32) - reciprocal.astype(np.int32))
        )
    ),
    hashes={
        name: sha(a.bundle / name)
        for name in ("results.json", "execution.json", "provenance.json")
    },
    reference_hashes={
        str(v.relative_to(ROOT)): sha(v)
        for v in (
            Path(__file__),
            ROOT / "lib/Numerics/sdk_math_reference.py",
            ROOT / "lib/Numerics/binary16.py",
        )
    },
)
a.output.write_text(json.dumps(out, indent=2) + "\n")
print(a.output)
print({k: v for k, v in out.items() if k not in ("hashes", "reference_hashes")})
