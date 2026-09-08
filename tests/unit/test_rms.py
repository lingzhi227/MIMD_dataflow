
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

import copy, json, sys, unittest
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from frontend import parse, Error
from mesh_rms import verify, plan, reference
from pragma_contracts import parse as pragma
from rms_debug import inspect


class RMS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(ROOT / "benchmarks/inference/waferllm/rmsnorm_64x128_8x8/hls.cpp")
        cls.m = verify(cls.raw, 6, 1)

    def test_fail_closed_policy_shape_and_precision(self):
        for change in (
            lambda m: m["nodes"][1].update(shape=[128, 1]),
            lambda m: m["nodes"][2].update(epsilon=0.0),
            lambda m: m["nodes"][2].update(epsilon=1e-12),
            lambda m: m["nodes"][2]["dataflow"].update(weights="rows"),
            lambda m: m["nodes"][2]["dataflow"].update(accumulation="f32"),
            lambda m: m["nodes"][2]["dataflow"].update(cols=3),
            lambda m: m["nodes"][0].update(dtype="f32"),
        ):
            m = copy.deepcopy(self.raw)
            change(m)
            with self.assertRaises((Error, KeyError)):
                verify(m, 6, 1)

    def test_feature_weight_and_row_factor_are_different_axes(self):
        s = plan(self.m)
        x = (
            np.tile(np.asarray([-0.5, 0.25, 0.5, -0.25]), (64, 32))
            * ((1 + np.arange(64) % 8) / 8)[:, None]
        )
        w = ((np.arange(128) % 17 - 8) / 8).reshape(1, 128)
        local, total, inv, y = reference(s, x, w)
        nominal = x * w / np.sqrt(np.mean(x * x, axis=1)[:, None] + 1e-6)
        self.assertLess(np.linalg.norm(y - nominal) / np.linalg.norm(nominal), 0.002)
        self.assertEqual(local.shape, (8, 8, 8))
        self.assertTrue(np.array_equal(total[:, 0], total[:, 7]))
        self.assertFalse(np.array_equal(inv[0, 0, :1], inv[0, 0, 1:2]))

    def test_half_overflow_rejected(self):
        s = plan(self.m)
        with self.assertRaises(Error):
            reference(s, np.full((64, 128), 100.0), np.ones((1, 128)))

    def test_pragma_order_and_debug_limits(self):
        text = "#pragma csl dataflow " + " ".join(
            f"{k}={v}"
            for k, v in reversed(list(self.raw["nodes"][2]["dataflow"].items()))
        )
        self.assertIn("weights=feature_columns", pragma(text))
        s = plan(self.m)
        self.assertEqual(inspect(s, None, "p7_3", 0, 3)["logical_features"], [112, 128])
        for node, epoch, step in [("p8_0", 0, 0), ("p0_0", 6, 0), ("p0_0", 0, 4)]:
            with self.assertRaises(Error):
                inspect(s, None, node, epoch, step)


if __name__ == "__main__":
    unittest.main()
