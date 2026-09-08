import copy, sys, tempfile, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from frontend import parse, Error
from ir import verify
from mesh_pair_rotation import plan, generate
from mesh_pair_rotation_sdk import pack_features, unpack
from pair_rotation_debug import inspect


class BatchedPair(unittest.TestCase):
    def raw(self):
        return parse(
            ROOT / "projects/waferllm/batched_pair_rotation_5x1024_8x8_x/hls.cpp"
        )

    def test_axes_batch_major_every_word_and_replica(self):
        for axis in ("x", "y"):
            r = self.raw()
            r["nodes"][3]["dataflow"]["axis"] = axis
            s = plan(verify(r, 6, 8))
            a = np.arange(5 * 1024).reshape(5, 1024)
            packed = pack_features(s, a)
            for y in range(8):
                for x in range(8):
                    k = x if axis == "x" else y
                    np.testing.assert_array_equal(
                        packed[y, x].reshape(5, 128), a[:, k * 128 : (k + 1) * 128]
                    )
            np.testing.assert_array_equal(unpack(s, packed), a)
            self.assertEqual(s["resources"]["explicit_dsr"], [1, 2, 3, 4, 5])
            self.assertEqual(s["memory_per_pe"]["scratch_bytes"], 512)
            d = inspect(s, None, "p7_3", 0, 0)
            self.assertEqual(d["logical_rows"], [0, 5])
            self.assertEqual(d["tile_order"], "batch-major")

    def test_illegal_layout_or_split_pairs(self):
        for change in (
            lambda r: r["nodes"][3]["dataflow"].update(coefficients="per_token"),
            lambda r: r["nodes"][3]["dataflow"].update(axis="z"),
            lambda r: r["nodes"][3]["dataflow"].update(cols=3),
            lambda r: r["nodes"][3]["dataflow"].update(compute="dsd"),
        ):
            r = self.raw()
            change(r)
            with self.assertRaises(Error):
                verify(r, 6, 8)

    def test_legacy_generated_csl_unchanged(self):
        p = ROOT / "projects/waferllm/pair_rotation_64x128_8x8_broadcast_odd_even"
        s = plan(verify(parse(p / "hls.cpp"), 6, 8))
        prior = ROOT / "tests/fixtures/history/legacy-pair-generated"
        with tempfile.TemporaryDirectory() as td:
            generate(s, td)
            for f in Path(td).iterdir():
                self.assertEqual(f.read_bytes(), (prior / f.name).read_bytes())

    def test_actual_completed_prefix_and_false_full_claims(self):
        import json
        from mesh_pair_rotation_sdk import audit_cases

        root = (
            ROOT
            / "tests/fixtures/history/run-20260907T230157971748Z"
        )
        read = lambda n: json.loads((root / n).read_text())
        s, m, bs, r = [
            read(n)
            for n in ("schedule.json", "semantic.json", "batches.json", "results.json")
        ]
        prefix = dict(
            r,
            success=False,
            cases=r["cases"][:2],
            diagnostics=r["diagnostics"][:2],
            launches=r["launches"][:2],
        )
        report = audit_cases(s, m, bs, prefix, require_complete=False)
        self.assertEqual(report["epochs"], 2)
        self.assertFalse(report["full_run_passed"])
        for bad in (dict(prefix, success=True), dict(prefix, runtime_instances=True)):
            with self.assertRaises(Error):
                audit_cases(s, m, bs, bad, require_complete=False)
        with self.assertRaises(Error):
            audit_cases(s, m, bs, prefix)
        corrupted = copy.deepcopy(prefix)
        corrupted["diagnostics"][1]["result"][7][7][-1] ^= 1
        with self.assertRaises(AssertionError):
            audit_cases(s, m, bs, corrupted, require_complete=False)

    def test_debug_missing_call_explicit(self):
        s = plan(verify(self.raw(), 6, 8))
        with self.assertRaisesRegex(Error, "not observed"):
            inspect(s, {"diagnostics": []}, "p0_0", 1, 0)


if __name__ == "__main__":
    unittest.main()
