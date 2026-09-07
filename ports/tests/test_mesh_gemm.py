import copy
import json
import math
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from frontend import parse, Error
from ir import verify
from planner import plan
from mesh_common import pack_tiles, unpack_tiles, validate_sdk_options
from roundoff import check_matrix_roundoff


class SummaContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(
            ROOT / "projects/sdk_examples/mesh_gemm_64x64x64_4x4_vector/hls.cpp"
        )

    def test_join_and_physical_placement(self):
        s = plan(verify(self.raw, 4, 64))
        self.assertEqual(s["stages"][2]["after"], ["broadcast_A", "broadcast_B"])
        self.assertEqual(len(s["nodes"]), 16)
        self.assertEqual(s["nodes"][-1]["place"], [7, 4])

    def test_simulator_options_are_explicit_and_bounded(self):
        valid = {"suppress_trace": True, "num_threads": 16, "dump_core": True}
        validate_sdk_options(valid)
        for invalid in (
            dict(valid, num_threads=0),
            dict(valid, num_threads=True),
            dict(valid, suppress_trace="yes"),
            dict(valid, extra=1),
        ):
            with self.assertRaises(Error):
                validate_sdk_options(invalid)

    def test_tail_and_nonsquare_rejected(self):
        for rows, cols in ((3, 3), (4, 2)):
            m = copy.deepcopy(self.raw)
            m["nodes"][2]["dataflow"].update(rows=rows, cols=cols)
            with self.assertRaises(Error):
                verify(m, 4, 64)

    def test_tile_packing_golden_values(self):
        a = np.arange(96, dtype=np.float32).reshape(12, 8)
        f = pack_tiles(a, 3, 2, "F")
        self.assertEqual(
            f[0, 0].tolist(), [0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27]
        )
        self.assertEqual(f[2, 1, -1], 95)
        for order in ("C", "F"):
            np.testing.assert_array_equal(
                unpack_tiles(pack_tiles(a, 3, 2, order), 4, 4, order), a
            )

    def test_roundoff_rejects_missing_panel_and_wrong_tile(self):
        a = np.arange(1, 33, dtype=np.float32).reshape(4, 8) / 16
        b = np.arange(1, 33, dtype=np.float32).reshape(8, 4) / 16
        ref = a.astype(float) @ b.astype(float)
        for bad in (a[:, :4].astype(float) @ b[:4].astype(float), ref[::-1]):
            with self.assertRaisesRegex(ValueError, "envelope exceeded"):
                check_matrix_roundoff(a, b, bad, ref)

    def test_roundoff_rejects_stale_accumulator(self):
        a = np.zeros((4, 8), np.float32)
        b = np.ones((8, 4), np.float32)
        with self.assertRaisesRegex(ValueError, "envelope exceeded"):
            check_matrix_roundoff(a, b, np.ones((4, 4)), np.zeros((4, 4)))

    def test_nonfinite_overflow_and_non_f32_rejected(self):
        for value in (
            float("nan"),
            float("inf"),
            np.finfo(np.float32).max,
            0.1,
            2.0**-149,
        ):
            with self.assertRaises(ValueError):
                check_matrix_roundoff([[value]], [[2]], [[0]], [[0]])

    def test_cancellation_accuracy_screen_is_not_hidden(self):
        # The input sum is mathematically 1; sequential f32 can lose it.
        a = np.array([[2.0**24, 1, -(2.0**24)]], np.float32)
        b = np.ones((3, 1), np.float32)
        report = check_matrix_roundoff(a, b, [[0]], [[1]])
        self.assertFalse(report["old_fixed_tolerance_passed"])
        self.assertLessEqual(report["max_error_over_bound"], 1)

    def test_original_random_failure_is_preserved(self):
        fixture = json.loads(
            (ROOT / "tests/fixtures/summa-original-random.json").read_text()
        )
        a = np.asarray(fixture["inputs"]["a"]).reshape(128, 256)
        b = np.asarray(fixture["inputs"]["b"]).reshape(256, 128)
        witness = fixture["witness"]
        row, col = witness["row"], witness["column"]
        reference = math.fsum(float(a[row, k]) * float(b[k, col]) for k in range(256))
        self.assertEqual(reference, witness["fsum"])
        report = check_matrix_roundoff(
            a[row : row + 1],
            b[:, col : col + 1],
            [[witness["source_f32"]]],
            [[reference]],
        )
        self.assertFalse(report["old_fixed_tolerance_passed"])
        self.assertLess(report["max_error_over_bound"], 1)


if __name__ == "__main__":
    unittest.main()
