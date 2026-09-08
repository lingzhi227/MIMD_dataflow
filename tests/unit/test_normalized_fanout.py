"""Undispatched fan-out semantic planning: shared producer and resource bounds."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, sys, unittest
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from frontend import parse, Error
from mesh_normalized_fanout import verify, plan


class NormalizedFanout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_raw = parse(ROOT / "benchmarks/inference/waferllm/normalized_matmul_64x128_8x8/hls.cpp")

    def raw(self, count):
        m = copy.deepcopy(self.__class__.base_raw)
        base = m["nodes"][3:]
        m["nodes"] = m["nodes"][:3]
        for i in range(count):
            q, mm, out = copy.deepcopy(base)
            q.update(id="weight" + str(i), host="weight" + str(i))
            mm.update(id="projection" + str(i))
            mm["inputs"][1] = q["id"]
            out.update(id="sink" + str(i), host="output" + str(i), inputs=[mm["id"]])
            m["nodes"] += [q, mm, out]
        return m

    def test_shared_alignment_and_no_extra_resources(self):
        s = plan(verify(self.raw(3), 6, 1))
        self.assertEqual(s["projections"], 3)
        self.assertEqual(len(s["stages"]), 5)
        self.assertFalse(s["ownership_transition"]["extra_copy"])
        self.assertEqual(s["resources"]["colors"], list(range(1, 12)))

    def test_reject_alias_or_independent_normalization(self):
        m = self.raw(3)
        m["nodes"][7]["inputs"][0] = m["nodes"][0]["id"]
        with self.assertRaises(Error):
            verify(m, 6, 1)
        m = self.raw(3)
        m["nodes"][7]["inputs"][1] = m["nodes"][3]["id"]
        with self.assertRaises(Error):
            verify(m, 6, 1)

    def test_three_large_sampled_rejects_but_counter_fits(self):
        m = self.raw(3)
        for n in m["nodes"]:
            if n["id"] == "w":
                n["shape"] = [1, 256]
            elif n["op"] == "input" and n["id"].startswith("weight"):
                n["shape"] = [256, 256]
            elif n["op"] != "output":
                n["shape"] = [128, 256]
        with self.assertRaises(Error):
            verify(m, 6, 1)
        m["instrumentation"] = "counters"
        s = plan(verify(m, 6, 1))
        self.assertLess(sum(s["memory_per_pe"].values()), 49152)
