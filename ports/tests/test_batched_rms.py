"""Feature ownership, odd batches, range/resource rejection and padding semantics."""

import copy, sys, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from frontend import parse, Error
from ir import verify
from mesh_batched_rms import plan, reference, inputs
from mesh_batched_rms_sdk import packed, decode
from batched_rms_fixtures import batches, check


class BatchedRMS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(ROOT / "projects/waferllm/batched_rms_3x512_8x8_g2/hls.cpp")
        cls.m = verify(cls.raw, 8, 2)
        cls.s = plan(cls.m)

    def test_odd_batch_does_not_partition_over_mesh(self):
        s = self.s
        self.assertEqual((s["B"], s["padded_batches"], s["Nt"]), (3, 4, 64))
        self.assertEqual(s["B"] % s["P"], 3)
        self.assertEqual(s["resources"]["colors"], [5, 6, 7, 8, 9])
        self.assertIn("not mesh-wide", s["resources"]["join"])
        self.assertLess(sum(s["memory_per_pe"].values()), 49152)

    def test_shape_policy_and_range_rejections(self):
        for change in (
            lambda m: m["nodes"][2]["dataflow"].update(axis="x"),
            lambda m: m["nodes"][2]["dataflow"].update(layout="column_major"),
            lambda m: m["nodes"][2]["dataflow"].update(groups=3),
            lambda m: m["nodes"][2]["dataflow"].update(reconfigure="after_send"),
            lambda m: m["nodes"][1].update(shape=[3, 512]),
            lambda m: m["nodes"][2].update(epsilon=1e-12),
            lambda m: m["nodes"][0].update(dtype="f32"),
        ):
            value = copy.deepcopy(self.raw)
            change(value)
            with self.assertRaises((Error, KeyError)):
                verify(value, 8, 2)
        value = copy.deepcopy(self.raw)
        for i in (0, 2):
            value["nodes"][i]["shape"] = [16, 2048]
        value["nodes"][1]["shape"] = [1, 2048]
        value["nodes"][2]["dataflow"].update(rows=4, cols=4)
        with self.assertRaisesRegex(Error, "memory"):
            verify(value, 8, 2)

    def test_all_eight_target_gates_replicas_and_padding(self):
        for b in batches(3, 512):
            x, w = inputs(self.m, b)
            local, total, inv, target = reference(self.s, x, w)
            self.assertTrue(np.all(local[:, -1] == 0))
            self.assertEqual(total[-1], 0)
            mapped = packed(self.s, self.m, b)
            for y in range(8):
                np.testing.assert_array_equal(
                    mapped["X"][y, 0].reshape(3, 64), x[:, y * 64 : (y + 1) * 64]
                )
                for column in range(1, 8):
                    np.testing.assert_array_equal(
                        mapped["X"][y, column], mapped["X"][y, 0]
                    )
            check(3, 512, b, dict(normalized=target.ravel().tolist()))


if __name__ == "__main__":
    unittest.main()
