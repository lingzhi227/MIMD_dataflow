"""Development mixed plan: role typing, native interpretation and CSL widths."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, json, sys, tempfile, unittest
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT / "lib"), str(ROOT / "experiments")]
from frontend import parse, Error
from input_attention_mixed_source import source
from mesh_input_attention_mixed import verify, plan, generate, evaluate
from input_attention_mixed_lifetimes import validate


class MixedPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "hls.cpp"
            p.write_text(source())
            cls.raw = parse(p)
        cls.raw["instrumentation"] = "counters"
        cls.m = verify(cls.raw, 8, 2)
        cls.s = plan(cls.m)

    def test_types_memory_and_old_range_removal(self):
        s = self.s
        self.assertEqual(len(s["precision_boundaries"]), 7)
        from mesh_input_attention_mixed_sdk import extents, WIDE_PORTS

        ext = extents(s)
        self.assertEqual(ext["mixed_v"], 64)
        self.assertEqual(ext["wide_probability"], 64)
        self.assertTrue(WIDE_PORTS <= set(ext))
        self.assertTrue(s["qualification"]["admitted"])
        self.assertFalse(s["qualification"]["catalog_qualified"])
        self.assertLess(sum(s["memory_per_pe"].values()), 49152)
        for k in (
            "normalization_bounds",
            "prelude_numerical_bounds",
            "attention_output_bound",
            "attention_schedule",
        ):
            self.assertNotIn(k, s)
        self.assertEqual(s["storage_lifetimes"]["storage"]["mixed_v"], 4 * 64)
        self.assertEqual(s["storage_lifetimes"]["validation"]["phases"], 25)
        self.assertTrue(validate(s["storage_lifetimes"])["checked"])

    def test_wrong_narrowing_and_schedule_rejected(self):
        m = copy.deepcopy(self.raw)
        v = next(n for n in m["nodes"] if n["id"] == "v")
        v["dtype"] = v["precision"]["storage"] = "f16"
        with self.assertRaisesRegex(Error, "precision boundaries"):
            verify(m, 8, 2)
        m = copy.deepcopy(self.raw)
        m["instrumentation"] = "sampled"
        with self.assertRaises(Error):
            verify(m, 8, 2)

    def test_bad_alias_dtype_and_transport_rejected(self):
        life = copy.deepcopy(self.s["storage_lifetimes"])
        v = next(n for n in life["values"] if n["name"] == "mixed_v")
        v["storage_dtype"] = "f16"
        with self.assertRaisesRegex(Error, "storage dtype"):
            validate(life)
        life = copy.deepcopy(self.s["storage_lifetimes"])
        next(n for n in life["values"] if n["name"] == "mixed_a")["storage"] = "mixed_v"
        with self.assertRaisesRegex(Error, "alias"):
            validate(life)
        s = copy.deepcopy(self.s)
        s["typed_transport"][1]["left"] = "f16"
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(Error, "transport widths"):
                generate(s, t)

    def test_native_all_eight_and_observers_exact(self):
        from native_transport import parse_outputs
        from input_attention_mixed_reference import stages

        p = ROOT / "tests/fixtures/history/input-attention-mixed-frontend-20260907T134515612938Z"
        bs = json.loads((p / "batches.json").read_text())
        # expf may differ by an ulp across host libm implementations. Compare
        # with C++ actually executed on this host, not another host's stdout.
        import subprocess
        from host_compiler import executable

        self.assertEqual((p / "hls.cpp").read_text(), source())
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "observed"
            command = [
                executable(),
                "-std=c++17",
                "-ffp-contract=off",
                "-DMW_BOUND=2",
                "-DMW_EPOCHS=8",
                "-DMW_MAX_INPUT=16384",
                "-Werror",
                "-Wno-unknown-pragmas",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
                "-I",
                str(ROOT / "include/pragma"),
                str(p / "observed.cpp"),
                str(ROOT / "runtime/native/native.cpp"),
                "-o",
                str(binary),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            run = subprocess.run(
                [str(binary)],
                check=True,
                capture_output=True,
                text=True,
                input=(p / "native-input.txt").read_text(),
            )
            rows = parse_outputs(run.stdout)
        self.assertEqual(len(rows), len(bs))
        actual, _ = evaluate(self.m, bs)
        for b, row, result in zip(bs, rows, actual):
            self.assertEqual(result["output"], row["output"])
            for name, value in stages(self.m, b).items():
                if "__observed_" + name in row:
                    np.testing.assert_array_equal(
                        value.ravel(), row["__observed_" + name]
                    )

    def test_codegen_matches_executed_twelve_csl_files(self):
        old = ROOT / "tests/fixtures/history/input-attention-codegen-20260907T133613178311Z"
        with tempfile.TemporaryDirectory() as t:
            generate(self.s, t)
            files = list(Path(t).glob("*.csl"))
            self.assertEqual(len(files), 12)
            for p in files:
                target = next(
                    q for q in old.rglob(p.name) if "implementation" not in q.parts
                )
                self.assertEqual(p.read_bytes(), target.read_bytes(), p.name)
            s = copy.deepcopy(self.s)
            s["scale"] = 0.13
            generate(s, t)
            self.assertIn("@as(f32,0.13)", (Path(t) / "pe.csl").read_text())
            s = copy.deepcopy(self.s)
            s["epsilon"] = 2e-6
            generate(s, t)
            self.assertIn(".epsilon=@as(f32,2e-06)", (Path(t) / "pe.csl").read_text())


if __name__ == "__main__":
    unittest.main()
