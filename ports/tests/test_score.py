"""Transpose-view semantics, bounds and topology-sensitive numerical validation."""

import copy, sys, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from frontend import parse, Error
from mesh_score import verify, plan, reference
from score_debug import inspect
from score_fixtures import batches, check


class Score(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(ROOT / "projects/waferllm/score_64x128_8x8/hls.cpp")

    def test_typed_view_and_unchanged_input_ownership(self):
        s = plan(verify(self.raw, 6, 1))
        self.assertFalse(s["transpose"]["materialized"])
        self.assertEqual(s["resources"]["microthreads"], [2, 3])
        self.assertNotIn(7, s["resources"]["active_input_queues"])

    def test_wrong_view_edge_and_policy_rejected(self):
        for mode in ("edge", "order", "dtype"):
            m = copy.deepcopy(self.raw)
            if mode == "edge":
                m["nodes"][3]["inputs"].reverse()
            elif mode == "order":
                m["nodes"][3]["dataflow"]["order"] = "west_first"
            else:
                m["nodes"][2]["dtype"] = "f32"
            with self.assertRaises(Error):
                verify(m, 6, 1)

    def test_reordered_independent_inputs_canonicalized(self):
        m = copy.deepcopy(self.raw)
        m["nodes"][:2] = m["nodes"][1::-1]
        self.assertEqual(verify(m, 6, 1), verify(self.raw, 6, 1))

    def test_memory_mode_and_packing_fail_closed(self):
        m = copy.deepcopy(self.raw)
        for n in m["nodes"]:
            if n["op"] == "input":
                n["shape"] = [256, 256]
            elif n["op"] in ("matmul", "transpose"):
                n["shape"] = [256, 256]
        m["nodes"][3]["dataflow"].update(rows=4, cols=4)
        with self.assertRaises(Error):
            verify(m, 6, 1)
        m = copy.deepcopy(self.raw)
        m["nodes"][0]["shape"][0] = 65
        with self.assertRaises(Error):
            verify(m, 6, 1)

    def test_distinct_dense_inputs_and_corruption(self):
        bs = batches(8, 16)
        q = np.asarray(bs[4]["q"]).reshape(8, 16)
        k = np.asarray(bs[4]["k"]).reshape(8, 16)
        s = dict(P=4, Mt=2, Nt=4)
        parts, owners, roots, out = reference(s, q, k)
        self.assertEqual(parts.shape, (4, 4, 16))
        self.assertTrue(all(sorted(roots[y, 0]) == list(range(4)) for y in range(4)))
        self.assertTrue(
            check(8, 16, bs[4], dict(score=out.ravel().tolist()))[
                "fixed_accuracy_passed"
            ]
        )
        with self.assertRaises(AssertionError):
            check(8, 16, bs[4], dict(score=(out + 1).ravel().tolist()))

    def test_future_epoch_is_not_observed(self):
        s = plan(verify(self.raw, 6, 1))
        d = inspect(s, {"diagnostics": []}, "p0_0", 1, 0)
        self.assertFalse(d["observed"])
        self.assertIsNone(d["half_bits"])


if __name__ == "__main__":
    unittest.main()
