"""Keep next-boundary source semantics explicit without claiming a backend."""

import sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "toolchain"), str(ROOT / "experiments")]
from input_attention_source import source
from frontend import parse, Error
from ir import verify


class InputAttentionProposal(unittest.TestCase):
    def test_next_prefix_range_and_source_adapter_contract(self):
        import json
        from rms_projection_bounds import projection
        from pair_rotation_bounds import bound

        r = projection(0.125, 1.5, 0.00390625, 8, 8, 1e-6)
        self.assertLess(bound(r["projection_absolute"], 1, 1)["output_absolute"], 1)
        root = ROOT / "tests/fixtures/history/input-attention-source-20260907T122338297913Z"
        schema = json.loads((root / "schema.json").read_text())
        self.assertEqual(len(schema["inputs"]), 11)
        self.assertNotIn("q", schema["inputs"])
        source = (root / "prefill.csl").read_text()
        for i in range(1, 5):
            self.assertIn(f"|i|{{seq_len_p_pe}} -> X_tmp_{i}[i]", source)
        self.assertIn("hls_phase=-12", source)
        self.assertIn(
            "rms_local.normalize(ptr_X,ptr_W,ptr_X_norm,ptr_local_sum)", source
        )
        self.assertIn(
            "rms_local.normalize(ptr_Z,ptr_W,ptr_Z_norm,ptr_local_sum)", source
        )
        self.assertFalse(
            json.loads((root / "provenance.json").read_text())[
                "resident_supplied_qkv_tail"
            ]
        )
        self.assertTrue(
            json.loads((root / "provenance.json").read_text())["resident_input_prefix"]
        )

    def test_shared_source_operands_and_explicit_pair_order(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hls.cpp"
            p.write_text(source())
            m = parse(p)
        ns = m["nodes"]
        self.assertEqual(len(ns), 31)
        self.assertEqual(sum(n["op"] == "input" for n in ns), 11)
        norms = [n for n in ns if n["op"] == "rmsnorm"]
        self.assertEqual(norms[0]["inputs"][1], norms[1]["inputs"][1])
        residual = [n for n in ns if n["op"] == "add"][0]
        self.assertIn(norms[0]["inputs"][0], residual["inputs"])
        pairs = [n for n in ns if n["op"] == "rotate_pairs"]
        self.assertEqual(len(pairs), 2)
        self.assertTrue(
            all(
                n["pair_order"] == "odd_even"
                and n["dataflow"]["coefficients"] == "feature_pairs"
                for n in pairs
            )
        )
        self.assertEqual(pairs[0]["inputs"][1:], pairs[1]["inputs"][1:])
        with self.assertRaises(Error):
            verify(m, 8, 2)


if __name__ == "__main__":
    unittest.main()
