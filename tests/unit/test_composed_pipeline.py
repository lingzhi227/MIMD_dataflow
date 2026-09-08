
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

import copy, json, sys, tempfile, unittest
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from frontend import parse, Error
from ir import verify
from planner import plan
from backend import generate
from projected_cache_ffn_codegen import extents
from projected_cache_ffn_debug import inspect


class ComposedPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(
            ROOT / "benchmarks/inference/waferllm/projected_cache_ffn_3x256x512x512_16x16/hls.cpp"
        )
        cls.m = verify(cls.raw, 8, 2)
        cls.s = plan(cls.m)

    def test_public_pipeline_profile_and_ports(self):
        s = self.s
        self.assertEqual(s["profile"], "mesh_projected_cache_ffn.v1")
        self.assertEqual((s["rows"], s["cols"]), (16, 16))
        self.assertEqual(s["output_bindings"]["result"]["physical"], "ffn_result")
        self.assertEqual(extents(s)["ffn_weights"], 1536)
        self.assertEqual(sum(s["memory_per_pe"].values()), 44030)

    def test_name_independent_codegen(self):
        m = copy.deepcopy(self.raw)
        rename = {n["id"]: "node" + str(i) for i, n in enumerate(m["nodes"])}
        for n in m["nodes"]:
            n["id"] = rename[n["id"]]
            n["inputs"] = [rename[i] for i in n["inputs"]]
        m["nodes"].reverse()
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            generate(self.s, a)
            generate(plan(verify(m, 8, 2)), b)
            self.assertEqual(
                {p.name: p.read_bytes() for p in a.glob("*.csl")},
                {p.name: p.read_bytes() for p in b.glob("*.csl")},
            )

    def test_unavailable_debug_does_not_invent_data(self):
        view = inspect(self.s, None, "p15_15", 7, 35)
        self.assertFalse(view["available"])
        self.assertIsNone(view["raw_words"])

    def test_unsupported_placement_and_counter_policy_rejected(self):
        m = copy.deepcopy(self.raw)
        next(n for n in m["nodes"] if n["id"] == "up")["place"] = [0, 0]
        with self.assertRaises(Error):
            verify(m, 8, 2)
        with self.assertRaises(Error):
            verify(dict(self.raw, instrumentation="counters"), 8, 2)
