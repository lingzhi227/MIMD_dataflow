
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
from frontend import parse
from ir import verify
from mesh_qr import factor, plan
from mesh_qr_sdk import check_factor, audit_rotations
from qr_schedule import rotations


class QRContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(ROOT / "benchmarks/linear_algebra/matrix_algorithms/mesh_qr_128x64_8x4/hls.cpp")

    def test_rectangular_mesh(self):
        s = plan(verify(self.raw, 4, 64))
        self.assertEqual((s["rows"], s["cols"], s["Nt"]), (8, 4, 16))
        self.assertEqual(len(s["nodes"]), 32)

    def test_unknown_instrumentation_rejected(self):
        m = verify(self.raw, 4, 64)
        m["instrumentation"] = "silent_unknown"
        with self.assertRaises(ValueError):
            plan(m)

    def test_unequal_local_tiles_rejected(self):
        m = copy.deepcopy(self.raw)
        m["nodes"][1]["dataflow"]["cols"] = 2
        with self.assertRaises(ValueError):
            verify(m, 4, 64)

    def test_givens_branches_and_negative_signs(self):
        for a in (
            np.diag([-2.0, 3.0]),
            np.array([[0.0, 2.0], [3.0, 1.0]]),
            np.array([[4.0, 1.0], [1.0, 3.0], [2.0, -1.0]]),
        ):
            r = np.asarray(factor(a.ravel(), *a.shape)).reshape(a.shape)
            check_factor(a, r)
            flipped = r.copy()
            flipped[0] *= -1
            check_factor(a, flipped)

    def test_rank_and_condition_rejected_before_sdk(self):
        for a in (np.ones((2, 2)), np.zeros((2, 2)), np.diag([1.0, 1e-5])):
            with self.assertRaises(ValueError):
                factor(a.ravel(), 2, 2)

    def test_gram_and_reference_reject_wrong_factor(self):
        a = np.array([[4.0, 1.0], [1.0, 3.0], [2.0, -1.0]])
        with self.assertRaises((ValueError, AssertionError)):
            check_factor(a, np.ones_like(a))

    def test_symbolic_small_geometry(self):
        self.assertEqual(
            [[len(rotations(2, 2, 2, x, y)) for x in range(2)] for y in range(2)],
            [[3, 3], [4, 5]],
        )
        self.assertEqual(rotations(2, 2, 2, 0, 0), [1, 2, 2])
        self.assertEqual(rotations(2, 2, 2, 1, 1), [1, 3, 1, 3, 1])

    def test_rotation_ring_and_corruption(self):
        w = np.zeros((16, 8))
        for serial in range(200):
            if serial < 7 or serial % 16 == 0:
                slot = serial if serial < 7 else 7 + (serial // 16) % 9
                w[slot] = [1, serial, 1, 0, 3, 4, 3, 4]
        self.assertEqual(audit_rotations(w.tolist(), 200)["samples"], 16)
        for column, value in ((1, 0.5), (2, 0), (6, 30), (0, 4)):
            bad = w.copy()
            bad[0, column] = value
            with self.assertRaises((ValueError, AssertionError)):
                audit_rotations(bad.tolist(), 200)


if __name__ == "__main__":
    unittest.main()
