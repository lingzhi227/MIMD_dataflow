"""Native f32 interpreter must preserve FMA, including double-rounding edges."""

import sys, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "toolchain"))
from float32_arithmetic import fma, primitive
from frontend import Error


class Float32Arithmetic(unittest.TestCase):
    def test_double_rounding_midpoint_and_scalar(self):
        a, b, c = map(np.float32, (64 + 2**-17, 1 - 2**-23, 2**30 + 128))
        self.assertNotEqual(np.float32(float(a) * float(b) + float(c)), c)
        self.assertEqual(fma(a, b, c), c)
        self.assertEqual(fma(np.array([a]), b, c)[0], c)

    def test_random_fma_matches_libm(self):
        rng = np.random.default_rng(64210)
        # Include widely separated exponents; avoid overflow in this finite test.
        values = [
            np.ldexp(
                rng.uniform(-1, 1, 5000).astype(np.float32),
                rng.integers(-120, 60, 5000),
            ).astype(np.float32)
            for _ in range(3)
        ]
        fn = primitive("fmaf", 3)
        expected = np.array([fn(*map(float, t)) for t in zip(*values)], np.float32)
        self.assertTrue(
            np.array_equal(fma(*values).view(np.uint32), expected.view(np.uint32))
        )

    def test_subnormal_ties_and_overflow_rejection(self):
        tiny = np.nextafter(np.float32(0), np.float32(1))
        self.assertEqual(fma(tiny, np.float32(0.5), np.float32(0)), 0)
        self.assertEqual(fma(tiny, np.float32(1.5), np.float32(0)), 2 * tiny)
        with self.assertRaisesRegex(Error, "overflow"):
            fma(np.finfo(np.float32).max, np.float32(2), np.float32(0))


if __name__ == "__main__":
    unittest.main()
