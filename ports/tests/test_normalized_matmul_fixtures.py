"""Independent standard composition detects row/feature ownership mistakes."""

import unittest
import numpy as np
from normalized_matmul_fixtures import batches, check


class ResidentFixture(unittest.TestCase):
    def test_identity_and_zero(self):
        b = batches(4, 8)
        for i in (0, 3, 4):
            x = np.asarray(b[i]["x"]).reshape(4, 8)
            w = np.asarray(b[i]["w"])
            q = np.asarray(b[i]["q"]).reshape(8, 8)
            y = (x * w / np.sqrt(np.mean(x * x, axis=1)[:, None] + 1e-6)) @ q
            check(
                4,
                8,
                b[i],
                {"projected": y.astype(np.float16).astype(float).ravel().tolist()},
            )

    def test_corrupted_result(self):
        b = batches(4, 8)[0]
        for v in (float("nan"), 0.0, 100.0):
            with self.assertRaises(AssertionError):
                check(4, 8, b, {"projected": [v] * 32})
