"""Pair order, coefficient ownership, shape safety and observed source semantics."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, sys, unittest
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from frontend import parse, Error
from ir import verify
from mesh_pair_rotation import plan, reference, accuracy
from mesh_pair_rotation_sdk import packed


class PairRotation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = parse(
            ROOT / "benchmarks/inference/waferllm/pair_rotation_64x128_8x8_token_even_odd/hls.cpp"
        )

    def test_explicit_order_identity_and_source_permutation(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        c = np.ones((1, 2))
        s = np.zeros((1, 2))
        np.testing.assert_array_equal(reference(x, c, s, "even_odd")[1], x)
        np.testing.assert_array_equal(reference(x, c, s, "odd_even")[1], [[2, 1, 4, 3]])

    def test_reject_split_pairs_and_wrong_coefficients(self):
        for change in [
            lambda m: m["nodes"][3]["dataflow"].update(cols=128),
            lambda m: m["nodes"][1].update(shape=[1, 64]),
            lambda m: m["nodes"][3].update(pair_order="implicit"),
            lambda m: m["nodes"][3].update(inputs=["x", "cosine", "cosine"]),
        ]:
            m = copy.deepcopy(self.raw)
            change(m)
            with self.assertRaises(Error):
                verify(m, 6, 8)

    def test_declaration_order_is_canonical(self):
        m = copy.deepcopy(self.raw)
        m["nodes"][:3] = m["nodes"][:3][::-1]
        self.assertEqual(plan(verify(m, 6, 8)), plan(verify(self.raw, 6, 8)))

    def test_row_scratch_and_broadcast_ownership(self):
        m = copy.deepcopy(self.raw)
        m["nodes"][3]["dataflow"].update(rows=4, coefficients="feature_pairs")
        for n in m["nodes"][1:3]:
            n["shape"] = [1, 64]
        s = plan(verify(m, 6, 8))
        self.assertEqual(s["memory_per_pe"]["scratch_bytes"], 8 * 16)
        self.assertEqual(s["coefficient_length"], 8)
        values = packed(
            s, [np.zeros((64, 128)), np.arange(64).reshape(1, 64), np.zeros((1, 64))]
        )
        np.testing.assert_array_equal(values["cosine"][0], values["cosine"][3])
        np.testing.assert_array_equal(values["cosine"][0, 2], np.arange(16, 24))

    def test_cancellation_accuracy_does_not_use_output_relative_error(self):
        x = np.full((2, 4), 8.0)
        c = np.full((2, 2), 0.5)
        s = c.copy()
        target = reference(x, c, s, "even_odd")[1]
        self.assertTrue(accuracy(x, c, s, "even_odd", target)["fixed_accuracy_passed"])
        with self.assertRaises(Error):
            accuracy(x, c, s, "even_odd", target + 1)
