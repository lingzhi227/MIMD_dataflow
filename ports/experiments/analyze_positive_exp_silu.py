"""Full finite-half magnitude exp/SiLU observation, including underflow boundaries."""

import argparse, json, math
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify
from positive_exp_silu_candidate import evaluate
from binary16 import bits

p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
verify(a.bundle)
execution = read(a.bundle / "execution.json")
r = read(a.bundle / "results.json")
assert (
    execution["success"]
    and execution["results_sha256"] == sha(a.bundle / "results.json")
    and r["success"]
    and len(r["cases"]) == 2
)
cases = []
for epoch, c in enumerate(r["cases"]):
    raw = np.arange(0x7C00, dtype=np.uint16)
    if epoch:
        raw = raw[::-1]
    np.testing.assert_array_equal(np.asarray(c["input"], np.uint16).ravel(), raw)
    np.testing.assert_array_equal(c["progress"], epoch + 1)
    x = raw.view(np.float16).astype(float)
    expected = np.asarray([[bits(v) for v in evaluate(float(t))] for t in x], np.uint16)
    observations = {}
    for i, name in enumerate(
        ("value", "negative_silu", "stable_negative_silu", "positive_silu")
    ):
        v = np.asarray(c[name])
        assert (
            v.shape == (8, 3968)
            and np.issubdtype(v.dtype, np.integer)
            and np.all((v >= 0) & (v < 65536))
        )
        v = v.astype(np.uint16).ravel()
        np.testing.assert_array_equal(v, expected[:, i])
        observations[name] = v.view(np.float16).astype(float)
    # Stable standard double expression keeps representable negative half tails.
    t = np.exp(-x)
    refneg = -x * t / (1 + t)
    refpos = x / (1 + t)
    negative = observations["negative_silu"]
    stable = observations["stable_negative_silu"]
    bounded = x <= 8
    cases.append(
        dict(
            exact_compound_half_bits=True,
            positive_exp_infinities=int(np.isinf(observations["value"]).sum()),
            negative_silu_zero_but_nearest_nonzero=int(
                np.count_nonzero((negative == 0) & (refneg.astype(np.float16) != 0))
            ),
            stable_half_zero_but_nearest_nonzero=int(
                np.count_nonzero((stable == 0) & (refneg.astype(np.float16) != 0))
            ),
            negative_silu_max_abs_error=float(np.max(np.abs(negative - refneg))),
            stable_negative_max_abs_error=float(np.max(np.abs(stable - refneg))),
            bounded_0_8_negative_relative_l2=float(
                np.linalg.norm((negative - refneg)[bounded])
                / np.linalg.norm(refneg[bounded])
            ),
            bounded_0_8_positive_relative_l2=float(
                np.linalg.norm((observations["positive_silu"] - refpos)[bounded])
                / np.linalg.norm(refpos[bounded])
            ),
        )
    )
a.output.write_text(
    json.dumps(
        dict(
            passed=True,
            new_sdk_execution=False,
            scope="All nonnegative finite half magnitudes, original SiLU at both signs; target bits and standard tail differences. This is math/source observation, not an HLS application qualification.",
            cases=cases,
            hashes={
                str(v): sha(v)
                for v in [
                    a.bundle / "provenance.json",
                    a.bundle / "results.json",
                    Path(__file__),
                    Path(__file__).with_name("positive_exp_silu_candidate.py"),
                    Path(__file__).with_name("half_exp_candidate.py"),
                ]
            },
        ),
        indent=2,
    )
    + "\n"
)
print(a.output)
print(cases)
