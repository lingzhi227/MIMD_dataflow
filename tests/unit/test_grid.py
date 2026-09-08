
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
from grid_ir import verify
from grid_plan import plan, validate_resources
from vectorize import ordered_terms


class GridContracts(unittest.TestCase):
    def module(self):
        return verify(
            parse(ROOT / "benchmarks/stencil/sdk_examples/resident_stencil_2x2x32_t4/hls.cpp"),
            2,
            64,
        )

    def test_ordered_vector_proof(self):
        m = self.module()
        self.assertEqual(
            ordered_terms(m["nodes"][2]["body"], 32),
            [[6, 6], [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
        )

    def test_actual_neighbor_topology(self):
        s = plan(self.module())
        self.assertEqual(len(s["nodes"]), 4)
        self.assertEqual(s["nodes"][0]["neighbors"], {"east": "p1_0", "north": "p0_1"})
        self.assertEqual(sum(len(n["neighbors"]) for n in s["nodes"]), 8)

    def test_microthread_conflict_is_rejected(self):
        resources = copy.deepcopy(plan(self.module())["nodes"][0]["resources"])
        resources["send_microthread"] = resources["neighbor_receive_microthreads"][0]
        with self.assertRaisesRegex(Error, "ownership overlap"):
            validate_resources(resources)

    def test_trace_memory_budget(self):
        m = self.module()
        m["nodes"][2]["grid"]["steps"] = 128
        m["epochs"] = 16
        with self.assertRaisesRegex(Error, "memory budget"):
            plan(m)

    def test_reject_mutated_vector_output(self):
        k = copy.deepcopy(self.module()["nodes"][2]["body"])
        k["body"][1][1][3][1][-1][1][2] = ["const", "0", "int"]
        with self.assertRaisesRegex(Error, "duplicate output"):
            ordered_terms(k, 32)


if __name__ == "__main__":
    unittest.main()
