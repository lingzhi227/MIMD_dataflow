"""Structural composition and lifetime regression for the supplied-attention tail."""

import copy, json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "toolchain"), str(ROOT / "experiments")]
from frontend import parse, Error
from build_prefill_tail import source
from mesh_prefill_tail import verify, plan
from region_lifetimes import verify as verify_lifetimes


class PrefillTail(unittest.TestCase):
    def module(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hls.cpp"
            p.write_text(source())
            return parse(p)

    def test_structural_composition(self):
        m = self.module()
        ids = {n["id"]: f"v{i}" for i, n in enumerate(m["nodes"])}
        for n in m["nodes"]:
            n["id"] = ids[n["id"]]
            n["inputs"] = [ids[i] for i in n["inputs"]]
            if n["op"] == "add":
                n["inputs"].reverse()
        s = plan(verify(m, 8, 2))
        self.assertEqual(s["profile"], "mesh_prefill_tail.v1")
        self.assertFalse(s["composition"]["intermediate_host_transfer"])
        self.assertEqual(len(s["storage_lifetimes"]["phases"]), 11)
        self.assertLess(sum(s["memory_per_pe"].values()), 49152)

    def test_reject_original_residual_at_final_add(self):
        m = self.module()
        final = [n for n in m["nodes"] if n["op"] == "add"][-1]
        final["inputs"][0] = next(
            n["id"] for n in m["nodes"] if n.get("host") == "residual"
        )
        with self.assertRaises(Error):
            verify(m, 8, 2)

    def test_reject_live_z_reuse_and_missing_join(self):
        r = plan(verify(self.module(), 8, 2))["storage_lifetimes"]
        bad = copy.deepcopy(r)
        v = next(v for v in bad["values"] if v["name"] == "projection_then_live_Z")
        bad["values"].append(dict(v, name="illegal_reuse", first=8, last=8))
        with self.assertRaises(Error):
            verify_lifetimes(bad["storage"], bad["values"], bad["phases"])
        bad = copy.deepcopy(r)
        bad["phases"][0]["release"] = []
        with self.assertRaises(Error):
            verify_lifetimes(bad["storage"], bad["values"], bad["phases"])

    def test_debugger_does_not_invent_counter_history(self):
        from prefill_tail_debug import inspect

        s = plan(verify(self.module(), 8, 2))
        s["instrumentation"] = "counters"
        tile = lambda v: [[[v] for _ in range(8)] for _ in range(8)]
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
                    "projection_snapshot",
                    "post_projection_z",
                    "prelude_progress",
                )
            )
        }
        r = dict(diagnostics=[d])
        self.assertFalse(inspect(s, r, "p0_0", 0, 0)["observed"])
        self.assertEqual(inspect(s, r, "p0_0", 0, 8)["half_bits"], [10])
        self.assertEqual(inspect(s, r, "p0_0", 0, 9)["half_bits"], [11])
        self.assertEqual(inspect(s, r, "p0_0", 0, 10)["stage"], "normalized")
        self.assertEqual(inspect(s, r, "p0_0", 0, 37)["half_bits"], [5])
        with self.assertRaises(Error):
            inspect(s, r, "p0_0", 0, 38)

    def test_summary_requires_internal_projection_and_delta(self):
        sys.path.insert(0, str(ROOT))
        from run_ports import numerical_summary

        row = dict(
            contract="supplied-attention-tail-half-normwise-v1",
            fixed_accuracy_passed=True,
        )
        for keys in ((), ("projection",), ("mlp_delta",)):
            value = dict(row, **{k: dict(fixed_accuracy_passed=True) for k in keys})
            with self.assertRaises(ValueError):
                numerical_summary(dict(native_application_checks=[value]))
        row.update(
            projection=dict(fixed_accuracy_passed=True),
            mlp_delta=dict(fixed_accuracy_passed=True),
        )
        self.assertTrue(
            numerical_summary(dict(native_application_checks=[row]))[
                "fixed_accuracy_passed"
            ]
        )

    def test_debugger_before_first_saved_call(self):
        import subprocess

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "schedule.json").write_text(
                json.dumps(dict(profile="mesh_prefill_tail.v1"))
            )
            raw = subprocess.check_output(
                [
                    sys.executable,
                    str(ROOT / "toolchain/debug.py"),
                    td,
                    "--check-completed",
                ],
                text=True,
            )
            report = json.loads(raw)
            self.assertFalse(report["completed_call_diagnostic"]["available"])
            self.assertEqual(report["completed_call_diagnostic"]["completed_calls"], 0)
            self.assertFalse(report["diagnostic_is_full_qualification"])

    def test_existing_ffn_codegen_unchanged(self):
        from mesh_feed_forward import generate

        old = (
            ROOT
            / "tests/fixtures/history/run-20260907T090013451671Z"
        )
        schedule = json.loads((old / "schedule.json").read_text())
        with tempfile.TemporaryDirectory() as td:
            generate(schedule, Path(td))
            for p in Path(td).rglob("*.csl"):
                self.assertEqual(
                    p.read_bytes(),
                    (old / p.relative_to(td)).read_bytes(),
                    str(p),
                )


if __name__ == "__main__":
    unittest.main()
