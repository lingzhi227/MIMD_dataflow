
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

import copy
from pathlib import Path
import sys
import unittest

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from frontend import parse, Error
from ir import verify
from planner import plan


class MeshGemvContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(
            ROOT / "benchmarks/linear_algebra/sdk_examples/mesh_gemv_64x64_4x4_vector/hls.cpp"
        )

    def test_explicit_dataflow_reaches_schedule(self):
        s = plan(verify(self.raw, 4, 64))
        self.assertEqual(
            (s["kernel_rows"], s["kernel_cols"], s["Mt"], s["Nt"]), (4, 4, 16, 16)
        )
        self.assertEqual(s["compute"], "vector")
        self.assertEqual(len(s["stages"]), 6)

    def test_rejects_unimplemented_tail(self):
        m = copy.deepcopy(self.raw)
        m["nodes"][0]["shape"][0] = 66
        m["nodes"][2]["shape"][0] = 66
        with self.assertRaisesRegex(Error, "evenly divisible"):
            verify(m, 4, 64)

    def test_large_global_tensor_is_partitioned_not_local(self):
        m = copy.deepcopy(self.raw)
        m["nodes"][0]["shape"] = [512, 512]
        m["nodes"][1]["shape"] = [512, 1]
        m["nodes"][2]["shape"] = [512, 1]
        m["nodes"][2]["dataflow"].update(rows=2, cols=2)
        with self.assertRaisesRegex(Error, "memory budget"):
            verify(m, 4, 64)
        m["nodes"][2]["dataflow"].update(rows=8, cols=8)
        self.assertLess(
            sum(plan(verify(m, 4, 64))["memory_per_pe"].values()), 48 * 1024
        )

    def test_no_silent_numerical_policy_change(self):
        m = copy.deepcopy(self.raw)
        m["nodes"][2]["dataflow"]["fp"] = "strict"
        with self.assertRaisesRegex(Error, "numerical/dataflow policy"):
            verify(m, 4, 64)


if __name__ == "__main__":
    unittest.main()
