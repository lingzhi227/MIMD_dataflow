"""Recheck the shared production model against both exhaustive SDK probes."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from sdk_math_reference import exp_f16, exp_f16_nonpositive, silu_f16
from binary16 import bits

p = argparse.ArgumentParser()
p.add_argument("negative", type=Path)
p.add_argument("positive", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
words = 0
for bundle in (a.negative, a.positive):
    verify(bundle)
    e = read(bundle / "execution.json")
    assert e["success"] and e["results_sha256"] == sha(bundle / "results.json")
    for c in read(bundle / "results.json")["cases"]:
        raw = np.asarray(c["input"], np.uint16).ravel()
        x = raw.view(np.float16).astype(float)
        np.testing.assert_array_equal(
            np.asarray(c["value"], np.uint16).ravel(), [bits(exp_f16(v)) for v in x]
        )
        words += len(x)
        if bundle == a.negative:
            np.testing.assert_array_equal(
                [bits(exp_f16_nonpositive(v)) for v in x],
                np.asarray(c["value"], np.uint16).ravel(),
            )
        else:
            for sign, key in [(-1, "negative_silu"), (1, "positive_silu")]:
                np.testing.assert_array_equal(
                    np.asarray(c[key], np.uint16).ravel(),
                    [bits(silu_f16(sign * v)) for v in x],
                )
                words += len(x)
a.output.write_text(
    json.dumps(
        dict(
            passed=True,
            new_sdk_execution=False,
            observed_half_words=words,
            scope="Production signed finite-half exp and original half SiLU at both signs; nonpositive API retained. Numerical tail limitations remain explicit.",
            hashes={
                str(v): sha(v)
                for v in [
                    ROOT / "lib/Numerics/sdk_math_reference.py",
                    a.negative / "results.json",
                    a.positive / "results.json",
                    Path(__file__),
                ]
            },
        ),
        indent=2,
    )
    + "\n"
)
print(a.output, words)
