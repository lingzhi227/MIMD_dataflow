import copy, sys, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from frontend import parse, Error
from mesh_softmax import verify, plan, reference, inputs, source_text


class Softmax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(ROOT / "projects/waferllm/softmax_64x128_8x8/hls.cpp")
        cls.m = verify(cls.raw, 6, 1024)

    def test_allnegative_source_failure_is_finite_uniform(self):
        h, e, y = reference(plan(self.m), np.full((64, 128), -1024.0))
        np.testing.assert_array_equal(y, 1 / 128)
        np.testing.assert_array_equal(e, 1)
        np.testing.assert_array_equal(h[:, :, 1, :], -128)

    def test_distinct_feature_peaks(self):
        x = np.full((64, 128), -128.0)
        j = (np.arange(64) * 17 + 3) % 128
        x[np.arange(64), j] = 128
        _, _, y = reference(plan(self.m), x)
        np.testing.assert_array_equal(np.argmax(y, axis=1), j)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertLess(float(np.max(abs(y.sum(axis=1) - 1))), 0.001)

    def test_optional_map_is_an_explicit_numerical_preserving_strategy(self):
        raw = copy.deepcopy(self.raw)
        raw["nodes"][1]["dataflow"]["elementwise"] = "map"
        mapped = plan(verify(raw, 6, 1024))
        scalar = plan(self.m)
        self.assertEqual(mapped["rows"], scalar["rows"])
        self.assertEqual(mapped["resources"]["colors"], scalar["resources"]["colors"])
        self.assertEqual(source_text(mapped, "layout"), source_text(scalar, "layout"))
        self.assertIn("@map(hls_exp_value,yd,hls_exp_dsd)", source_text(mapped, "pe"))
        self.assertNotIn("@map(hls_exp_value", source_text(scalar, "pe"))
        raw["nodes"][1]["dataflow"]["elementwise"] = "automatic"
        with self.assertRaises(Error):
            verify(raw, 6, 1024)

    def test_fail_closed(self):
        for edit in [
            lambda m: m["nodes"][1].update(scale=0.0),
            lambda m: m["nodes"][1].update(scale=1e-12),
            lambda m: m["nodes"][1]["dataflow"].update(reduce="add"),
            lambda m: m["nodes"][1]["dataflow"].update(cols=3),
            lambda m: m["nodes"][1].pop("dataflow"),
        ]:
            m = copy.deepcopy(self.raw)
            edit(m)
            with self.assertRaises(Error):
                verify(m, 6, 1024)
        m = copy.deepcopy(self.raw)
        m["nodes"][1]["scale"] = 8.0
        with self.assertRaises(Error):
            verify(m, 6, 4096)
        with self.assertRaises(Error):
            inputs(self.m, {"x": ["0"] * 8192})


if __name__ == "__main__":
    unittest.main()
