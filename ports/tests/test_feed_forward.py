"""Source-bound FFN graph contracts and compatibility of the shared CSL engine."""

import copy, json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "toolchain"), str(ROOT / "experiments")]
from frontend import parse, Error
from build_feed_forward import source
from mesh_feed_forward import verify, plan, core
from mesh_mlp import verify as public_mlp_verify, generate as mlp_generate
from csl_region_hooks import render


class FeedForward(unittest.TestCase):
    def module(self, text=None):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hls.cpp"
            p.write_text(text or source())
            return parse(p)

    def test_shared_normalization_and_live_residual(self):
        m = verify(self.module(), 8, 2)
        s = plan(m)
        self.assertEqual(len(m["nodes"]), 13)
        self.assertLess(sum(s["memory_per_pe"].values()), 49152)
        self.assertLess(s["normalization_bounds"]["output"], 12.1)
        self.assertFalse(s["composition"]["intermediate_host_transfer"])
        self.assertGreater(core(m)["input_bound"], 1)
        with self.assertRaises(Error):
            public_mlp_verify(core(m), 8, 2)

    def test_commuted_add_and_renamed_ids(self):
        m = self.module()
        ids = {n["id"]: "node_" + str(i) for i, n in enumerate(m["nodes"])}
        for n in m["nodes"]:
            n["id"] = ids[n["id"]]
            n["inputs"] = [ids[i] for i in n["inputs"]]
        next(n for n in m["nodes"] if n["op"] == "add")["inputs"].reverse()
        self.assertEqual(verify(m, 8, 2)["profile"], "mesh_feed_forward.v1")

    def test_reject_wrong_residual_range_or_region(self):
        for mutate in (
            lambda m: next(n for n in m["nodes"] if n["op"] == "add")[
                "inputs"
            ].__setitem__(0, m["nodes"][1]["id"]),
            lambda m: m["nodes"][2].update(abs_bound=0.125),
            lambda m: next(n for n in m["nodes"] if n["op"] == "add")[
                "dataflow"
            ].update(rows=4),
        ):
            m = self.module()
            mutate(m)
            with self.assertRaises(Error):
                verify(m, 8, 2)
        text = (
            source()
            .replace(" accumulation=block_f32", "")
            .replace(
                "spatial::matmul_blocked<spatial::f16,spatial::scalar>(hidden,d,32)",
                "spatial::matmul(hidden,d)",
            )
        )
        with self.assertRaises(Error):
            verify(self.module(text), 8, 2)

    def test_debugger_observed_scope(self):
        from feed_forward_debug import inspect

        s = plan(verify(self.module(), 8, 2))
        s["instrumentation"] = "counters"
        tile = lambda value: [[[value] for _ in range(8)] for _ in range(8)]
        d = {
            k: tile(i)
            for i, k in enumerate(
                (
                    "progress",
                    "queues",
                    "timing",
                    "normalized",
                    "down_snapshot",
                    "result",
                    "rms_progress",
                    "wide_accumulator",
                    "up_accumulator",
                    "gate_accumulator",
                )
            )
        }
        results = dict(diagnostics=[d])
        self.assertEqual(inspect(s, results, "p0_0", 0, 0)["stage"], "normalized")
        self.assertFalse(inspect(s, results, "p0_0", 0, 1)["observed"])
        last_up = inspect(s, results, "p0_0", 0, 8)
        self.assertIsNone(last_up["half_bits"])
        self.assertEqual(last_up["f32_accumulator_bits"], [8])
        self.assertEqual(inspect(s, results, "p0_0", 0, 26)["half_bits"], [4])
        self.assertEqual(inspect(s, results, "p0_0", 0, 27)["half_bits"], [5])
        self.assertFalse(inspect(s, results, "p0_0", 1, 0)["available"])
        for step in (-1, 28):
            with self.assertRaises(Error):
                inspect(s, results, "p0_0", 0, step)

    def test_residual_cannot_hide_missing_delta_gate(self):
        sys.path.insert(0, str(ROOT))
        from run_ports import numerical_summary

        row = dict(
            contract="normalized-feed-forward-half-normwise-v1",
            fixed_accuracy_passed=True,
        )
        with self.assertRaisesRegex(ValueError, "separately observed"):
            numerical_summary(dict(native_application_checks=[row]))
        row["mlp_delta"] = dict(fixed_accuracy_passed=True)
        self.assertTrue(
            numerical_summary(dict(native_application_checks=[row]))[
                "fixed_accuracy_passed"
            ]
        )
        row["mlp_delta"]["fixed_accuracy_passed"] = False
        with self.assertRaises(ValueError):
            numerical_summary(dict(native_application_checks=[row]))

    def test_default_hooks_preserve_executed_mlp_csl(self):
        bundles = [
            ROOT / "tests/fixtures/history/run-20260907T044301583844Z",
            ROOT
            / "tests/fixtures/history/run-20260907T064549573557Z",
        ]
        for p in bundles:
            s = json.loads((p / "schedule.json").read_text())
            with tempfile.TemporaryDirectory() as td:
                mlp_generate(s, td)
                for f in Path(td).iterdir():
                    self.assertEqual(f.read_bytes(), (p / f.name).read_bytes(), f.name)
        with self.assertRaises(Error):
            render("no composition points")
        with self.assertRaises(Error):
            render("", dict(UNKNOWN="invalid"))


if __name__ == "__main__":
    unittest.main()
