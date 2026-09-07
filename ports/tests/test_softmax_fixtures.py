"""Independent probability checks reject numerically plausible protocol errors."""

import unittest
import numpy as np
from softmax_fixtures import batches, check


class SoftmaxFixtures(unittest.TestCase):
    def test_uniform_and_peaks(self):
        b = batches(8, 32)
        for index in (1, 2, 4):
            result = check(8, 32, b[index], {"probability": [1 / 32] * 256})
            self.assertEqual(result["max_row_mass_error"], 0)
        x = np.asarray(b[3]["x"]).reshape(8, 32)
        p = np.exp((x - x.max(axis=1)[:, None]) * 0.125)
        p /= p.sum(axis=1)[:, None]
        check(
            8,
            32,
            b[3],
            {"probability": p.astype(np.float16).astype(float).ravel().tolist()},
        )

    def test_reject_bad_mass_and_negative(self):
        b = batches(8, 32)[2]
        for value in (0.0, -1.0, float("nan"), 0.04):
            with self.assertRaises(AssertionError):
                check(8, 32, b, {"probability": [value] * 256})

    def test_wrong_row_ownership(self):
        b = batches(8, 32)[3]
        x = np.asarray(b["x"]).reshape(8, 32)
        p = np.zeros_like(x)
        p[np.arange(8), x.argmax(axis=1)] = 1
        with self.assertRaises(AssertionError):
            check(8, 32, b, {"probability": np.roll(p, 1, axis=0).ravel().tolist()})
