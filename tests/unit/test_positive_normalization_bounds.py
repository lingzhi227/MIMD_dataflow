
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

import math, sys, unittest
from pathlib import Path
from fractions import Fraction

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from positive_normalization_bounds import (
    probability_mass,
    weighted_contraction,
    half_ceiling,
)
from frontend import Error


class PositiveNormalizationBounds(unittest.TestCase):
    def test_outward_exact_fraction_and_half_rounding(self):
        r = probability_mass(512, 64, 8)
        exact = Fraction(r["exact_numerator"], r["exact_denominator"])
        self.assertGreaterEqual(Fraction.from_float(r["bound"]), exact)
        self.assertLess(
            Fraction.from_float(math.nextafter(r["bound"], -math.inf)), exact
        )
        self.assertGreater(r["bound"], 1)
        self.assertLess(r["bound"], 1.04)
        self.assertEqual(half_ceiling(Fraction(1, 2**25)), 2**-24)
        self.assertEqual(half_ceiling(Fraction(1)), 1)

    def test_global_mass_is_not_multiplied_by_participants(self):
        r = weighted_contraction(512, 64, 8, 32, 8.25)
        self.assertLess(r["local"], 9)
        self.assertLess(r["reduced"], 9)
        self.assertGreaterEqual(r["reduced"], r["local"])
        self.assertGreater(r["reduced"], 8.25)
        self.assertLess(
            weighted_contraction(512, 64, 8, 16, 8.25)["reduced"], r["reduced"]
        )
        self.assertGreater(
            weighted_contraction(512, 64, 8, 64, 8.25)["reduced"], r["reduced"]
        )

    def test_unsupported_math_domain_and_overflow_are_rejected(self):
        for args in [(1024, 128, 8), (512, 32, 8), (512, 64, 16), (512, 64, True)]:
            with self.assertRaises(Error):
                probability_mass(*args)
        for block, value in [(17, 1), (32, math.inf), (32, -1), (32, 65504)]:
            with self.assertRaises(Error):
                weighted_contraction(512, 64, 8, block, value)


if __name__ == "__main__":
    unittest.main()
