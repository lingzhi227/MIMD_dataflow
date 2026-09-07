"""Power result semantics remain distinct from residual-based solvers."""

import copy, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from frontend import parse
from ir import verify
from planner import plan
from resident_abi import schema


class PowerFrontend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(ROOT / "projects/sdk_examples/mesh_power_512_4x4/hls.cpp")

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
