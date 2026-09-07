"""Reduction semantics, tail ownership and tiny-norm accuracy boundaries."""

import copy, sys, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from frontend import parse
from mesh_reduction import verify, plan, inputs
from mesh_reduction_sdk import distribute, check_result


class Reductions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dot = parse(ROOT / "projects/sdk_examples/mesh_dot_8191_4x4/hls.cpp")
        cls.norm = parse(ROOT / "projects/sdk_examples/mesh_nrm2_17_8x8/hls.cpp")

    def test_partition_tail_and_empty(self):
        for raw in (self.dot, self.norm):
            m = verify(raw, 4, 8)
            s = plan(m)
            v = np.arange(s["N"], dtype=np.float32)
            tiles = distribute(v, s)
            seen = []
            for y in range(s["rows"]):
                for x in range(s["cols"]):
                    lo = (y * s["cols"] + x) * s["local_length"]
                    n = max(0, min(s["local_length"], s["N"] - lo))
                    seen.extend(tiles[y, x, :n])
                    np.testing.assert_array_equal(tiles[y, x, n:], 0)
            np.testing.assert_array_equal(seen, v)

    def test_reject_malformed_dataflow(self):
        for alter in ("policy", "extra", "shape", "port", "memory"):
            raw = copy.deepcopy(self.dot)
            if alter == "policy":
                raw["nodes"][-2]["dataflow"]["result"] = "root"
            elif alter == "extra":
                raw["nodes"][-2]["dataflow"]["router"] = "guess"
            elif alter == "shape":
                raw["nodes"][1]["shape"] = [8190, 1]
            elif alter == "port":
                raw["nodes"][1]["host"] = "x"
            else:
                for n in raw["nodes"][:2]:
                    n["shape"] = [262144, 1]
                raw["nodes"][-2]["dataflow"].update(rows=2, cols=2)
            with self.subTest(alter=alter), self.assertRaises(ValueError):
                verify(raw, 4, 8)

    def test_tiny_norm_cannot_pass_as_zero(self):
        x = np.array([1e-30, -2e-30], np.float32).tolist()
        with self.assertRaises(AssertionError):
            check_result("nrm2", [x], [0])
        check_result("nrm2", [x], [np.float32(np.sqrt(sum(float(v) ** 2 for v in x)))])
        check_result("nrm2", [[0, 0]], [0])

    def test_reject_invalid_input(self):
        m = verify(self.norm, 1, 8)
        for bad in (float("nan"), float("inf"), 9.0, 0.1):
            x = [0.0] * 17
            x[0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                inputs(m, {"x": x})


if __name__ == "__main__":
    unittest.main()
