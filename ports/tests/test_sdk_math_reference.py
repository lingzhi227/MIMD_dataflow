"""Fixed observed SDK bits, independent from model-generated expected values.

Evidence: rms-math-20260906T212318272243Z, all31744nonnegative finite inputs.
These selected regressions include inputs where correctly rounded sqrt differs.
"""

import math, struct, sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "toolchain"))
from sdk_math_reference import (
    sqrt_f16,
    rms_inverse_f16,
    exp_f16_nonpositive,
    exp_f16,
    silu_f16,
)
from binary16 import bits
from frontend import Error


def half(word):
    return struct.unpack("<e", struct.pack("<H", word))[0]


class SDKHalfMath(unittest.TestCase):
    def test_observed_non_nearest_results(self):
        for operand, result in [
            (0x0007, 0x114A),
            (0x041B, 0x200E),
            (0x3001, 0x35A8),
            (0x4001, 0x3DA8),
            (0x6001, 0x4DA8),
        ]:
            self.assertEqual(bits(sqrt_f16(half(operand))), result)
            self.assertNotEqual(bits(math.sqrt(half(operand))), result)

    def test_observed_positive_exp_and_silu_boundaries(self):
        for operand, exponent, negative, positive in [
            (0, 0x3C00, 0x8000, 0),
            (0x3C00, 0x4170, 0xB44D, 0x39D9),
            (0x4800, 0x69D2, 0x997F, 0x4800),
            (0x498B, 0x7BF7, 0x8991, 0x498B),
            (0x498C, 0x7C00, 0x8000, 0x498C),
            (0x4A00, 0x7C00, 0x8000, 0x4A00),
        ]:
            x = half(operand)
            self.assertEqual(bits(exp_f16(x)), exponent)
            self.assertEqual(bits(silu_f16(-x)), negative)
            self.assertEqual(bits(silu_f16(x)), positive)
        self.assertNotEqual(
            bits(half(0x4A00) * -math.exp(-12) / (1 + math.exp(-12))), 0x8000
        )
        for x in (math.inf, math.nan):
            with self.assertRaises(Error):
                silu_f16(x)

    def test_observed_exp_non_nearest_bits(self):
        for operand, result in [
            (0x9982, 0x3BFB),
            (0xAD43, 0x3B5F),
            (0xAFEC, 0x3B11),
            (0xBA0B, 0x3785),
            (0xC169, 0x2C47),
        ]:
            self.assertEqual(bits(exp_f16_nonpositive(half(operand))), result)
            self.assertNotEqual(bits(math.exp(half(operand))), result)
        for value in (1, math.inf, -math.inf, math.nan, -0.1):
            with self.assertRaises(Error):
                exp_f16_nonpositive(value)

    def test_invalid_domains_fail_closed(self):
        for value in [-1, math.inf, math.nan, 0.1]:
            with self.assertRaises(Error):
                sqrt_f16(value)
        for dimension in (0, -1, True, 2049):
            with self.assertRaises(Error):
                rms_inverse_f16(1, dimension)
        for epsilon in (0, -1, math.inf, math.nan, 1e-12):
            with self.assertRaises(Error):
                rms_inverse_f16(0, 64, epsilon)


if __name__ == "__main__":
    unittest.main()
