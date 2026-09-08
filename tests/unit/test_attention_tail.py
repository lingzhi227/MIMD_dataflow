"""Resident composition, resource rejection, and shared-backend compatibility."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, json, sys, tempfile, unittest
from pathlib import Path

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib"), str(ROOT / "experiments")]
from frontend import parse, Error
from ir import verify
from mesh_attention_tail import plan, generate
from build_attention_tail import source
from region_lifetimes import verify as verify_lifetimes


class AttentionTail(unittest.TestCase):
    def module(self, m=64, n=64, f=256, mode="counters"):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hls.cpp"
            path.write_text(source(m, n, f))
            module = parse(path)
        module["instrumentation"] = mode
        return module

    def test_structural_renaming_and_nine_input_binding(self):
        module = self.module()
        ids = {v["id"]: f"v{i}" for i, v in enumerate(module["nodes"])}
        for v in module["nodes"]:
            v["id"] = ids[v["id"]]
            v["inputs"] = [ids[x] for x in v["inputs"]]
            if v["op"] == "add":
                v["inputs"].reverse()
        s = plan(verify(module, 8, 2))
        self.assertEqual(s["profile"], "mesh_attention_tail.v1")
        self.assertEqual(s["input_bindings"]["q"], "x")
        self.assertEqual(len(s["input_bindings"]), 9)
        self.assertEqual(len(s["storage_lifetimes"]["phases"]), 16)
        self.assertFalse(s["composition"]["intermediate_host_transfer"])
        self.assertLess(sum(s["memory_per_pe"].values()), 49152)

    def test_reject_wrong_final_residual(self):
        m = self.module()
        final = [v for v in m["nodes"] if v["op"] == "add"][-1]
        final["inputs"][0] = next(
            v["id"] for v in m["nodes"] if v.get("host") == "residual"
        )
        with self.assertRaises(Error):
            verify(m, 8, 2)

    def test_memory_rejection_and_supported_sample(self):
        with self.assertRaises(Error):
            verify(self.module(mode="sampled"), 8, 2)
        s = plan(verify(self.module(f=128, mode="sampled"), 8, 2))
        self.assertLess(sum(s["memory_per_pe"].values()), 49152)
        self.assertLess(
            sum(plan(verify(self.module(m=128), 8, 2))["memory_per_pe"].values()), 49152
        )

    def test_reject_missing_attention_join(self):
        r = copy.deepcopy(plan(verify(self.module(), 8, 2))["storage_lifetimes"])
        r["phases"][0]["release"] = []
        with self.assertRaises(Error):
            verify_lifetimes(r["storage"], r["values"], r["phases"])

    def test_softmax_value_bound(self):
        from attention_output_bounds import bound

        for m in (64, 128, 512):
            r = bound(m, 8, 0.125)
            self.assertGreater(r["value_output_absolute"], 0.125)
            self.assertLess(r["denominator_upper"], 2**14)
            self.assertEqual(r["exp_certificate"]["operands"], 31745)
        self.assertEqual(bound(64, 8, 0)["value_output_absolute"], 0)
        with self.assertRaises(Error):
            bound(1024, 8, 0.125)

    def test_exports_and_phase_descriptor_reset(self):
        from mesh_attention_tail_sdk import extents

        s = plan(verify(self.module(), 8, 2))
        with tempfile.TemporaryDirectory() as td:
            generate(s, td)
            pe = (Path(td) / "pe.csl").read_text()
            layout = (Path(td) / "layout.csl").read_text()
            self.assertNotIn("HLS_REGION_", pe)
            for port in extents(s):
                self.assertIn(f'"{port}"', layout)
            self.assertIn(
                "if(phase==4){right_matrix_dsd=@increment_dsd_offset(right_matrix_dsd,1,f16);}",
                pe,
            )
            self.assertIn("attention_score_setup()", pe)
            self.assertEqual(pe.count("@bind_local_task(next_step,"), 1)

    def test_debugger_preserves_mandatory_observations(self):
        import numpy as np
        from attention_tail_debug import inspect
        from mesh_attention_tail_sdk import extents

        s = plan(verify(self.module(), 8, 2))
        d = {
            name: np.zeros((8, 8, length), dtype=np.uint16).tolist()
            for name, length in extents(s).items()
        }
        r = dict(diagnostics=[d])
        self.assertFalse(inspect(s, r, "p0_0", 0, 0)["observed"])
        for step, stage in (
            (8, "unscaled_qk"),
            (9, "softmax_probability"),
            (18, "resident_attention"),
            (56, "residual_output"),
        ):
            out = inspect(s, r, "p0_0", 0, step)
            self.assertTrue(out["observed"])
            self.assertEqual(out["stage"], stage)
        self.assertFalse(inspect(s, None, "p0_0", 0, 8)["available"])
        with self.assertRaises(Error):
            inspect(s, r, "p8_0", 0, 8)
        with self.assertRaises(Error):
            inspect(s, r, "p0_0", 0, 57)

    def test_summary_requires_all_five_branches(self):
        from run_profiles import numerical_summary

        keys = ("score", "probability", "attention", "projection", "mlp_delta")
        row = dict(
            contract="supplied-qkv-attention-tail-half-normwise-v1",
            fixed_accuracy_passed=True,
        )
        row.update({k: dict(fixed_accuracy_passed=True) for k in keys})
        for key in keys:
            bad = dict(row)
            bad.pop(key)
            with self.assertRaises(ValueError):
                numerical_summary(dict(native_application_checks=[bad]))
        self.assertTrue(
            numerical_summary(dict(native_application_checks=[row]))[
                "fixed_accuracy_passed"
            ]
        )

    def test_prior_tail_codegen_byte_identical(self):
        from mesh_prefill_tail import generate as old_generate

        old = (
            ROOT
            / "tests/fixtures/history/run-20260907T102005944852Z"
        )
        s = json.loads((old / "schedule.json").read_text())
        with tempfile.TemporaryDirectory() as td:
            old_generate(s, td)
            for p in Path(td).rglob("*.csl"):
                self.assertEqual(
                    p.read_bytes(), (old / p.relative_to(td)).read_bytes(), str(p)
                )


if __name__ == "__main__":
    unittest.main()
