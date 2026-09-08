
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

import copy
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from frontend import parse, Error
from ir import verify
from mesh_cholesky import factor_f32, plan
from mesh_cholesky_sdk import check_factor, checkpoint_reference
from mesh_common import pack_tiles


class CholeskyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(ROOT / "benchmarks/linear_algebra/sdk_examples/mesh_cholesky_32x32_4x4/hls.cpp")

    def test_triangular_schedule(self):
        s = plan(verify(self.raw, 4, 64))
        self.assertEqual(sum(n["active"] for n in s["nodes"]), 10)
        self.assertEqual(s["Nt"], 8)

    def test_nonsquare_and_tail_rejected(self):
        for rows, cols in ((4, 2), (3, 3)):
            m = copy.deepcopy(self.raw)
            m["nodes"][1]["dataflow"].update(rows=rows, cols=cols)
            with self.assertRaises(Error):
                verify(m, 4, 64)

    def test_spd_domain(self):
        for bad in ([1, 2, 2, 1], [1, 2, 0, 1], [1, 0, 0, float("nan")]):
            with self.assertRaises(ValueError):
                factor_f32(bad, 2)

    def test_factor_and_residual_reject_corruption(self):
        a = np.array([[4, 2], [2, 5]], dtype=float)
        good = np.array(factor_f32(a.ravel(), 2)).reshape(2, 2)
        check_factor(a, good)
        for bad in (good.T, good[::-1], good + np.eye(2) * 0.1):
            with self.assertRaises((ValueError, AssertionError)):
                check_factor(a, bad)

    def test_rank_one_checkpoint_golden_values(self):
        h = checkpoint_reference([[4, 2], [2, 5]], 2)
        np.testing.assert_array_equal(h[0,0], [[2,2],[0,0]])
        np.testing.assert_array_equal(h[1,0], [[1,1],[0,0]])
        np.testing.assert_array_equal(h[1,1], [[4,4],[2,2]])
        np.testing.assert_array_equal(h[0,1], np.zeros((2,2)))

    def test_row_major_golden_tile(self):
        tiles = pack_tiles(np.arange(64).reshape(8, 8), 2, 2, "C")
        self.assertEqual(
            tiles[1, 0].tolist(),
            [32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59],
        )


if __name__ == "__main__":
    unittest.main()
