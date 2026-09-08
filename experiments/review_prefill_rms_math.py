"""Revisit original RMS factors with the now exhaustively checked SDK model."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, verify, sha

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from sdk_math_reference import rms_inverse_f16
from binary16 import bits

p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
p.add_argument("model_qualification", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
verify(a.bundle)
q = read(a.model_qualification)
assert q["passed"] and q["source_sqrt_model_bits_exact"]
assert q["reference_hashes"]["lib/Numerics/sdk_math_reference.py"] == sha(
    ROOT / "lib/Numerics/sdk_math_reference.py"
)
e = read(a.bundle / "execution.json")
r = read(a.bundle / "results.json")
assert (
    e["success"]
    and e["results_sha256"] == sha(a.bundle / "results.json")
    and r["success"]
)
count = 0
for case in r["cases"]:
    reduced = np.asarray(case["hls_reduced"], np.uint16).view(np.float16)
    expected = np.asarray(
        [bits(rms_inverse_f16(float(v), 64)) for v in reduced.ravel()], np.uint16
    ).reshape(reduced.shape)
    np.testing.assert_array_equal(expected, case["hls_inverse"])
    count += expected.size
out = dict(
    passed=True,
    new_sdk_execution=False,
    exact_observed_inverse_words=count,
    scope="Original source RMS inverse factors now exactly explained by source-derived SDK sqrt + language half reciprocal; original feature-index error remains unchanged. Previous nearest-half mismatch evidence preserved.",
    hashes={
        str(p): sha(p)
        for p in [
            a.bundle / "results.json",
            a.model_qualification,
            ROOT / "lib/Numerics/sdk_math_reference.py",
            Path(__file__),
        ]
    },
)
a.output.write_text(json.dumps(out, indent=2) + "\n")
print(a.output, count)
