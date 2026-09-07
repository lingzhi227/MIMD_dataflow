"""Composition checks must preserve data edges, layout and phase-local ownership."""

import copy, sys, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from frontend import parse, Error
from mesh_score_softmax import verify, plan, evaluate
from score_softmax_fixtures import batches, check


class ScoreSoftmax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(ROOT / "projects/waferllm/score_softmax_64x128_8x8/hls.cpp")

    def test_same_region_and_completed_buffer_reuse(self):
        s = plan(verify(self.raw, 6, 1))
        self.assertFalse(s["ownership"]["intermediate_host_transfer"])
        self.assertEqual(s["resources"]["microthreads"], [2, 3])
        self.assertEqual(s["resources"]["colors"], list(range(1, 12)))

    def test_wrong_edge_or_different_region_rejected(self):
        for kind in ["edge", "region", "implementation"]:
            m = copy.deepcopy(self.raw)
            if kind == "edge":
                m["nodes"][4]["inputs"] = [m["nodes"][0]["id"]]
            elif kind == "region":
                m["nodes"][4]["dataflow"]["cols"] = 4
            else:
                m["nodes"][4]["dataflow"]["elementwise"] = "scalar"
            with self.assertRaises(Error):
                verify(m, 6, 1)

    def test_small_original_input_probability_reference(self):
        m = copy.deepcopy(self.raw)
        for node in m["nodes"]:
            if node["op"] == "input":
                node["shape"] = [8, 16]
            elif node["op"] == "transpose":
                node["shape"] = [16, 8]
            elif node["op"] in ("matmul", "softmax"):
                node["shape"] = [8, 8]
            if "dataflow" in node:
                node["dataflow"].update(rows=4, cols=4)
        m = verify(m, 6, 1)
        b = batches(8, 16)
        out, _ = evaluate(m, b)
        self.assertTrue(
            all(
                check(8, 16, m["nodes"][4]["scale"], x, y)["fixed_accuracy_passed"]
                for x, y in zip(b, out)
            )
        )
        with self.assertRaises(AssertionError):
            check(8, 16, m["nodes"][4]["scale"], b[0], {"probability": [0.0] * 64})


if __name__ == "__main__":
    unittest.main()
