"""Gated activation policy, dependency, tail and layout contracts."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, sys, unittest
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from frontend import parse, Error
from ir import verify
from mesh_swiglu import plan, inputs, reference, accuracy


class Gating(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(ROOT / "benchmarks/linear_algebra/waferllm/swiglu_64x256_8x8/hls.cpp")
        cls.m = verify(cls.raw, 6, 8)

    def test_reject_invalid_policy_and_edges(self):
        for mutate in [
            lambda m: m["nodes"][3].update(
                inputs=[m["nodes"][0]["id"], m["nodes"][1]["id"]]
            ),
            lambda m: m["nodes"][2]["dataflow"].update(math="standard"),
            lambda m: m["nodes"][3]["dataflow"].update(cols=4),
            lambda m: m["nodes"][1].update(dtype="f32"),
            lambda m: m["nodes"][0].update(shape=[32, 256]),
        ]:
            m = copy.deepcopy(self.raw)
            mutate(m)
            with self.assertRaises(Error):
                verify(m, 6, 8)
        with self.assertRaises(Error):
            verify(self.raw, 6, 12)

    def test_independent_declarations_and_commutative_product(self):
        m = copy.deepcopy(self.raw)
        m["nodes"][0], m["nodes"][1] = m["nodes"][1], m["nodes"][0]
        m["nodes"][3]["inputs"].reverse()
        self.assertEqual(plan(verify(m, 6, 8)), plan(self.m))

    def test_half_tail_allowance_is_not_relative_only(self):
        up = np.array([[8.0, -8.0, 1.0, 1.0]])
        gate = np.array([[-3 * 2**-24, 2**-24, 1.0, -1.0]])
        act, y = reference(up, gate)
        r = accuracy(up, gate, y)
        self.assertTrue(r["fixed_accuracy_passed"])
        self.assertLessEqual(r["max_error_over_allowance"], 1)
        self.assertEqual(int(np.float16(act[0, 2]).view(np.uint16)), 0x39D9)
        self.assertEqual(int(np.float16(act[0, 3]).view(np.uint16)), 0xB44D)
        with self.assertRaises(Error):
            accuracy(up, gate, y + 1)

    def test_odd_tile_and_memory_guard(self):
        m = copy.deepcopy(self.raw)
        for node in m["nodes"][:-1]:
            node["shape"] = [65, 129]
        for node in m["nodes"][2:4]:
            node["dataflow"].update(rows=5, cols=3)
        s = plan(verify(m, 6, 8))
        self.assertEqual(s["length"], 559)
        self.assertEqual(s["resources"]["colors"], [])
        for node in m["nodes"][:-1]:
            node["shape"] = [4096, 4096]
        for node in m["nodes"][2:4]:
            node["dataflow"].update(rows=16, cols=16)
        with self.assertRaises(Error):
            verify(m, 6, 8)
