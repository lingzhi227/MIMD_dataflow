"""Structured solver results and fail-closed separation from pending lowering."""

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
from frontend import parse
from ir import verify
from planner import plan


class SolverFrontend(unittest.TestCase):
    def test_record_fields_are_typed(self):
        m = verify(parse(ROOT / "benchmarks/linear_algebra/sdk_examples/mesh_cg_512_4x4/hls.cpp"), 4, 64)
        self.assertEqual(m["profile"], "mesh_cg.v1")
        outs = {n["host"]: n for n in m["nodes"] if n["op"] == "output"}
        self.assertEqual(outs["reason"]["dtype"], "u32")
        self.assertEqual(outs["iterations"]["dtype"], "u32")
        self.assertEqual(outs["residual_squared"]["shape"], [65, 1])
        s = plan(m)
        self.assertEqual(s["profile"], "mesh_cg.v1")
        self.assertLessEqual(s["estimated_bytes"], 49152)
        self.assertEqual(s["resources"]["collective_input_queues"], [3, 6, 5, 7])
        bad = copy.deepcopy(m)
        bad["nodes"][-1]["inputs"] = ["solved.solution"]
        with self.assertRaises(ValueError):
            verify(bad, 4, 64)

    def test_unsupported_resident_extents_and_capacity(self):
        from solver_ir import result_type

        raw = parse(ROOT / "benchmarks/linear_algebra/sdk_examples/mesh_cg_512_4x4/hls.cpp")
        for n, cap in [(256, 64), (512, 65)]:
            bad = copy.deepcopy(raw)
            bad["nodes"][2]["shape"] = [n + 1, 1]
            bad["nodes"][3]["shape"] = bad["nodes"][4]["shape"] = [n, 1]
            bad["nodes"][7]["result_type"] = result_type(n, cap)
            with self.subTest(n=n, capacity=cap), self.assertRaises(ValueError):
                plan(verify(bad, 4, 64))
        bad = copy.deepcopy(raw)
        bad["nodes"][7]["dataflow"]["rows_per_pe"] = 1
        with self.assertRaises(ValueError):
            plan(verify(bad, 4, 64))


if __name__ == "__main__":
    unittest.main()
