"""Analytical envelopes used by the resident SDK-plane FFN planner."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import sys, unittest
from pathlib import Path
from fractions import Fraction
import numpy as np

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from rms_l1_bounds import normalized_l1, projection, upward, half_ceiling
from frontend import Error


class CorrelatedRMS(unittest.TestCase):
    def test_outward_rounding(self):
        for v in (
            Fraction(1, 3),
            Fraction(2, 7),
            Fraction(123456789, 12345),
            Fraction(0),
        ):
            self.assertGreaterEqual(Fraction(upward(v)), v)
            self.assertGreaterEqual(Fraction(half_ceiling(upward(v))), v)

    def test_bound_and_missing_precondition_rejection(self):
        r = normalized_l1(1, 1, 256, 256, 32, 8, 1e-6)
        self.assertGreater(r["bound"], 256)
        self.assertLess(r["bound"], 259)
        q = projection(r["bound"], 0.03125, 32, 8)
        self.assertGreater(q["reduced"], 8)
        self.assertLess(q["reduced"], 8.25)
        for args in (
            (1, 1, 0, 256, 32, 8, 1e-6),
            (65504, 1, 256, 256, 32, 8, 1e-6),
            (1, 1, 256, 256, 32, 8, 0),
        ):
            with self.assertRaises(Error):
                normalized_l1(*args)

    def test_sampled_target_rows_are_inside_bound(self):
        from batched_ffn_reference import axis_sum, q
        from sdk_math_reference import rms_inverse_f16

        rng = np.random.default_rng(1579)
        bound = normalized_l1(1, 1, 256, 256, 32, 8, 1e-6)["bound"]
        rows = []
        for scale in (1, 2**-4, 2**-8, 2**-12, 2**-16, 2**-24):
            rows.extend(
                np.asarray(rng.uniform(-scale, scale, (8, 256)), np.float16).astype(
                    float
                )
            )
            rows.extend(np.vstack([np.full(256, scale), np.full(256, -scale)]))
        rows = np.array(rows)
        local = np.zeros((8, len(rows)))
        for y in range(8):
            for j in range(32):
                local[y] = q(local[y] + q(rows[:, 32 * y + j] ** 2))
        reduced = axis_sum(local, 0)
        inv = np.array([rms_inverse_f16(v, 256, 1e-6) for v in reduced])
        norm = q(rows * inv[:, None])
        self.assertTrue(np.all(np.sum(np.abs(norm), axis=1) <= bound))


if __name__ == "__main__":
    unittest.main()
