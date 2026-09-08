"""Composition validates edges, resource lifetimes and independent arithmetic."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, sys, unittest
from pathlib import Path

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from frontend import parse, Error
from mesh_attention import verify, plan, evaluate
from attention_debug import inspect
from attention_fixtures import batches, check


class Attention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(ROOT / "benchmarks/inference/waferllm/attention_64x128_8x8/hls.cpp")

    def test_lifetimes_and_input_order(self):
        m = verify(self.raw, 6, 1)
        s = plan(m)
        self.assertFalse(s["ownership"]["intermediate_host_transfer"])
        self.assertEqual(s["resources"]["microthreads"], [0, 1, 2, 3])
        self.assertIn("stride1", s["descriptor_entry_states"]["score_right"])
        shuffled = copy.deepcopy(self.raw)
        shuffled["nodes"][:3] = reversed(shuffled["nodes"][:3])
        self.assertEqual(verify(shuffled, 6, 1), m)
        self.assertFalse(inspect(s, None, "p3_2", 5, 2 * s["P"] + 3)["available"])

    def test_bad_edges_policies_and_memory(self):
        for kind in ("edge", "layout", "region", "memory"):
            m = copy.deepcopy(self.raw)
            if kind == "edge":
                m["nodes"][6]["inputs"][0] = m["nodes"][0]["id"]
            elif kind == "layout":
                m["nodes"][6]["dataflow"]["initial_align"] = "none"
            elif kind == "region":
                m["nodes"][6]["dataflow"].update(rows=4, cols=4)
            else:
                for n in m["nodes"]:
                    if n["shape"] is not None:
                        n["shape"] = [x * 2 for x in n["shape"]]
            with self.assertRaises(Error):
                verify(m, 6, 1)

    def test_small_independent_full_path(self):
        m = copy.deepcopy(self.raw)
        for n in m["nodes"]:
            if n["shape"] is not None:
                n["shape"] = [x // 8 for x in n["shape"]]
            if "dataflow" in n:
                n["dataflow"].update(rows=4, cols=4)
        m = verify(m, 6, 1)
        bs = batches(8, 16)
        out, _ = evaluate(m, bs)
        for b, o in zip(bs, out):
            self.assertTrue(
                check(8, 16, m["nodes"][5]["scale"], b, o)["fixed_accuracy_passed"]
            )
        with self.assertRaises(AssertionError):
            check(8, 16, m["nodes"][5]["scale"], bs[0], {"output": [0.0] * 128})

    def test_user_host_cannot_collide_with_internal_fragment(self):
        m = copy.deepcopy(self.raw)
        for n in m["nodes"]:
            if n["shape"] is not None:
                n["shape"] = [x // 8 for x in n["shape"]]
            if "dataflow" in n:
                n["dataflow"].update(rows=4, cols=4)
        baseline = verify(m, 6, 1)
        bs = batches(8, 16)
        expected, _ = evaluate(baseline, bs)
        m["nodes"][2]["host"] = "__probability"
        renamed = [dict(q=b["q"], k=b["k"], __probability=b["v"]) for b in bs]
        actual, _ = evaluate(verify(m, 6, 1), renamed)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
