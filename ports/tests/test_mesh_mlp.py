"""Typed MLP contracts: bounds, region, alias preconditions and frozen lowering."""

import copy, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain"), str(ROOT / "experiments")]
from frontend import parse, Error
from ir import verify
from mesh_mlp import plan
from build_mlp_profiles import source


class MLP(unittest.TestCase):
    def module(self, m=64, n=64, f=256):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hls.cpp"
            p.write_text(source(m, n, f, 8))
            return parse(p)

    def test_composition_bounds_and_storage(self):
        m = verify(self.module(), 6, 1)
        s = plan(m)
        self.assertEqual([v["K"] for v in s["projection_stages"]], [64, 64, 256])
        self.assertEqual(s["numerical_bounds"]["gate"], 1)
        self.assertFalse(s["ownership"]["intermediate_host_transfer"])
        self.assertLess(sum(s["memory_per_pe"].values()), 49152)

    def test_large_sample_memory_and_lean_contract(self):
        m = self.module(128, 128, 512)
        with self.assertRaisesRegex(Error, "memory"):
            verify(m, 6, 1)
        m["instrumentation"] = "counters"
        s = plan(verify(m, 6, 1))
        self.assertEqual(s["numerical_bounds"]["gate"], 2)
        self.assertEqual(s["numerical_bounds"]["hidden"], 4)

    def test_unproved_gate_rejected(self):
        m = self.module()
        for n in m["nodes"][:4]:
            n.pop("abs_bound")
        with self.assertRaisesRegex(Error, "SiLU domain"):
            verify(m, 6, 1)

    def test_shape_policy_and_shared_input_rejected(self):
        for mutate in [
            lambda m: m["nodes"][1].update(shape=[64, 128]),
            lambda m: m["nodes"][5]["dataflow"].update(rows=4),
            lambda m: m["nodes"][5].update(
                inputs=[m["nodes"][1]["id"], m["nodes"][2]["id"]]
            ),
        ]:
            m = self.module()
            mutate(m)
            with self.assertRaises(Error):
                verify(m, 6, 1)

    def test_ids_and_commuted_product_are_not_dispatch_keys(self):
        m = self.module()
        renames = {n["id"]: "renamed_" + n["id"] for n in m["nodes"]}
        for n in m["nodes"]:
            n["id"] = renames[n["id"]]
            n["inputs"] = [renames[i] for i in n["inputs"]]
        next(n for n in m["nodes"] if n["op"] == "multiply")["inputs"].reverse()
        self.assertEqual(verify(m, 6, 1)["profile"], "mesh_mlp.v1")

    def test_debugger_counter_tensors_are_unobserved(self):
        from mlp_debug import inspect

        m = self.module()
        m["instrumentation"] = "counters"
        s = plan(verify(m, 6, 1))
        d = {
            "progress": [[[0] * 8 for _ in range(8)] for _ in range(8)],
            "timing": [[[0] * 6 for _ in range(8)] for _ in range(8)],
            "queues": [[[248] * 2 for _ in range(8)] for _ in range(8)],
            "result": [[[0] * 64 for _ in range(8)] for _ in range(8)],
        }
        r = {"diagnostics": [d]}
        self.assertFalse(inspect(s, r, "p3_2", 0, 16)["observed"])
        self.assertTrue(inspect(s, r, "p3_2", 0, 25)["observed"])
        with self.assertRaises(Error):
            inspect(s, r, "p8_0", 0, 25)

    def test_small_independent_native_semantics(self):
        from mesh_mlp import evaluate
        from mlp_fixtures import batches, check

        m = verify(self.module(16, 16, 32), 6, 1)
        bs = batches(16, 16, 32)
        values, _ = evaluate(m, bs)
        for b, o in zip(bs, values):
            self.assertTrue(check(16, 16, 32, b, o)["fixed_accuracy_passed"])
        with self.assertRaises(AssertionError):
            check(16, 16, 32, bs[0], {"output": [0.0] * 256})


if __name__ == "__main__":
    unittest.main()
