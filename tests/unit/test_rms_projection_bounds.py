"""Range checks for the next resident input connection; no lowering qualification."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import sys, unittest
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from rms_projection_bounds import normalized_l1, projection, outward_half
from frontend import Error


class RMSProjectionBounds(unittest.TestCase):
    def test_correlated_bound_beats_component_product(self):
        from rms_bounds import row_norm_bound

        r = projection(0.125, 1.5, 0.00390625, 8, 8, 1e-6)
        self.assertLess(r["projection_absolute"], 0.4)
        old = row_norm_bound(0.125, 1.5, 8, 8, 1e-6)["output"] * 64 * 0.00390625
        self.assertGreater(old, 2.9)
        self.assertGreaterEqual(r["normalized_row"]["row_l1_upper"], 96)

    def test_zero_and_overflow_rejection(self):
        self.assertEqual(projection(0, 1.5, 0.1, 8, 8, 1e-6)["projection_absolute"], 0)
        self.assertEqual(
            projection(0.125, 0, 0.1, 8, 8, 1e-6)["projection_absolute"], 0
        )
        self.assertEqual(
            projection(0.125, 1.5, 0, 8, 8, 1e-6)["projection_absolute"], 0
        )
        with self.assertRaises(Error):
            outward_half(65503)
        with self.assertRaises(Error):
            normalized_l1(0.125, 1.5, 128, 8, 1e-6)

    def test_underflow_sparse_and_uniform_rows(self):
        from mesh_rms import reference
        from binary16 import matmul

        s = dict(rows=8, cols=8, Mt=8, Nt=8, M=64, N=64, epsilon=1e-6)
        x = np.zeros((64, 64), float)
        for i in range(44):
            magnitude = float(np.float16(2.0 ** (-24 + i % 22)))
            x[i, :] = magnitude
            if i >= 22:
                x[i, 1:] = 0
        x[44:] = 0.125
        g = np.full((1, 64), 1.5)
        cert = projection(0.125, 1.5, 0.00390625, 8, 8, 1e-6)
        target = reference(s, x, g)[-1]
        native = (
            (x * g / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-6))
            .astype(np.float16)
            .astype(float)
        )
        for value in (target, native):
            self.assertLessEqual(
                float(np.abs(value).sum(axis=1).max()),
                cert["normalized_row"]["row_l1_upper"],
            )
            product = matmul(value, np.full((64, 1), 0.00390625))
            self.assertLessEqual(
                float(np.abs(product).max()), cert["projection_absolute"]
            )

    def test_actual_qualified_sdk_rows_obey_certificate(self):
        import json
        from mesh_common import unpack_tiles

        base = (
            ROOT
            / "tests/fixtures/history/run-20260907T090013451671Z"
        )
        result = json.loads((base / "results.json").read_text())
        upper = normalized_l1(0.125, 1.5, 8, 8, 1e-6)["row_l1_upper"]
        for row in result["diagnostics"]:
            x = unpack_tiles(
                np.asarray(row["normalized"], np.uint16).view(np.float16), 8, 8, "F"
            ).astype(float)
            self.assertLessEqual(float(np.max(np.sum(np.abs(x), axis=1))), upper)


if __name__ == "__main__":
    unittest.main()
