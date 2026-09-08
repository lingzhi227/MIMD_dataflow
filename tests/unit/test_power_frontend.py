"""Power result semantics remain distinct from residual-based solvers."""

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
from resident_abi import schema


class PowerFrontend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(ROOT / "benchmarks/linear_algebra/sdk_examples/mesh_power_512_4x4/hls.cpp")

    def test_completion_record_and_abi(self):
        m = verify(self.raw, 8, 64)
        abi = schema(plan(m))
        self.assertEqual(m["profile"], "mesh_power.v1")
        self.assertEqual(abi["vector_inputs"], [["cg_solution", 3]])
        self.assertNotIn("true_residual_norm", abi["result_fields"])
        self.assertNotIn("cg_rhs", abi["symbols"])
        self.assertEqual(abi["symbols"]["cg_history"]["length"], 33)
        self.assertEqual(m["nodes"][-1]["shape"], [32, 1])

    def test_invalid_result_or_unsafe_workspace(self):
        for kind in ("record", "capacity", "policy"):
            bad = copy.deepcopy(self.raw)
            if kind == "record":
                bad["nodes"][-1]["inputs"] = [bad["nodes"][5]["id"] + ".vector"]
            elif kind == "capacity":
                bad["nodes"][5]["dataflow"]["rows_per_pe"] = 1
            else:
                bad["nodes"][5]["dataflow"]["fp"] = "ordered"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                plan(verify(bad, 8, 64))


if __name__ == "__main__":
    unittest.main()
