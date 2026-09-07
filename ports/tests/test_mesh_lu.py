import copy
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from frontend import parse, Error
from ir import verify
from mesh_lu import factor, plan
from mesh_lu_sdk import check_factor, checkpoint_reference


class LUContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(ROOT / "projects/matrix_algorithms/mesh_lu_32x32_4x4/hls.cpp")

    def test_distributed_schedule(self):
        s = plan(verify(self.raw, 4, 64))
        self.assertEqual(len(s["nodes"]), 16)
        self.assertEqual(s["resources"]["data_task_input_queues"], [4, 5])

    def test_no_implicit_pivoting_or_tail(self):
        for change in ({"pivot": "partial"}, {"rows": 3, "cols": 3}):
            raw = copy.deepcopy(self.raw)
            raw["nodes"][1]["dataflow"].update(change)
            with self.assertRaises(ValueError):
                verify(raw, 4, 64)

    def test_domain_and_breakdown_before_sdk(self):
        for bad in ([0, 1, 1, 0], [1, 2, 2, 1], [1, 0, 0, float("inf")]):
            with self.assertRaises(ValueError):
                factor(bad, 2)
        # Nonsymmetric input is valid in LU, unlike Cholesky.
        self.assertEqual(factor([4, 1, 2, 3], 2), [4, 1, 0.5, 2.5])

    def test_packed_factor_and_missing_update(self):
        a = np.array([[4, 1], [2, 3]], float)
        good = np.array(factor(a.ravel(), 2)).reshape(2, 2)
        check_factor(a, good)
        for bad in (a, good.T, good + np.eye(2) * 0.1):
            with self.assertRaises((ValueError, AssertionError)):
                check_factor(a, bad)

    def test_global_pivot_checkpoint(self):
        h = checkpoint_reference([[4, 1], [2, 3]], 2)
        np.testing.assert_array_equal(h[1, 1], [[2.5, 2.5], [2.5, 2.5]])
        np.testing.assert_array_equal(h[1, 0], [[0.5, 0.5], [0, 0]])


if __name__ == "__main__":
    unittest.main()
