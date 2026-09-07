"""Logical-tile contraction, ownership and diagnostic boundary regressions."""

import copy, sys, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from frontend import parse, Error
from mesh_device_matmul import verify, plan, reference
from device_matmul_debug import inspect
from device_matmul_fixtures import batches, check


class DeviceMatmul(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(ROOT / "projects/waferllm/device_matmul_64x128_8x8/hls.cpp")

    def test_layout_resources_and_independent_input_order(self):
        m = verify(self.raw, 6, 1)
        s = plan(m)
        self.assertEqual(s["resources"]["active_input_queues"], [5, 7])
        self.assertEqual(s["resources"]["microthreads"], [0, 1, 2, 3])
        r = copy.deepcopy(self.raw)
        r["nodes"][:2] = r["nodes"][1::-1]
        self.assertEqual(verify(r, 6, 1), m)

    def test_invalid_shapes_policy_and_memory(self):
        for kind in ("shape", "policy", "memory"):
            m = copy.deepcopy(self.raw)
            if kind == "shape":
                m["nodes"][0]["shape"][1] = 32
            elif kind == "policy":
                m["nodes"][2]["dataflow"]["initial_align"] = "forward"
            else:
                for n in m["nodes"]:
                    if n["op"] != "output":
                        n["shape"] = [256, 256]
                m["nodes"][2]["dataflow"].update(rows=4, cols=4)
            with self.assertRaises(Error):
                verify(m, 6, 1)

    def test_original_input_math_and_nontrivial_block_ownership(self):
        b = batches(8, 16)[1]
        a = np.asarray(b["a"]).reshape(8, 8)
        v = np.asarray(b["b"]).reshape(8, 16)
        s = dict(P=4, Mt=2, Nt=4)
        hist, left, right, out = reference(s, a, v)
        self.assertEqual(left.shape, (4, 4, 16))
        self.assertEqual(right.shape, (4, 4, 32))
        self.assertTrue(
            check(8, 16, b, {"product": out.ravel().tolist()})["fixed_accuracy_passed"]
        )
        with self.assertRaises(AssertionError):
            check(8, 16, b, {"product": (out + 1).ravel().tolist()})

    def test_future_epoch_not_observed(self):
        s = plan(verify(self.raw, 6, 1))
        d = inspect(s, {"diagnostics": []}, "p3_2", 1, 0)
        self.assertFalse(d["observed"])
        self.assertIsNone(d["half_bits"])


if __name__ == "__main__":
    unittest.main()
