"""A maximum repair must never change the independent sum initializer."""

import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from score_softmax_source_probe import repair_softmax_maximum
from attention_layout_adapter import align_column_major_value


class SourceSoftmaxAdapter(unittest.TestCase):
    def test_only_maximum_initializer_changes(self):
        src = (ROOT / "projects/waferllm/upstream/Prefill/src/prefill.csl").read_text()
        body = src[src.index("fn softmax_score()") : src.index("fn output_matmul()")]
        actual = repair_softmax_maximum(body)
        changed = [
            (a, b) for a, b in zip(body.splitlines(), actual.splitlines()) if a != b
        ]
        self.assertEqual(len(changed), 1)
        self.assertIn("-65504.0", changed[0][1])
        self.assertEqual(actual.count("@fmovh(comp_dest_dsr_1, 0.0); // Clearing"), 1)

    def test_repair_fails_closed_on_changed_source(self):
        with self.assertRaises(AssertionError):
            repair_softmax_maximum("fn softmax_score() void {}")

    def test_device_adapter_preserves_verified_source_functions(self):
        src = (ROOT / "projects/waferllm/upstream/Prefill/src/prefill.csl").read_text()
        actual = align_column_major_value(src)
        verified = (
            ROOT / "tests/fixtures/verified-attention-prefill.csl"
        ).read_text()
        for start, end in (
            ("fn output_matmul()", "fn h1_matmul()"),
            ("task right_matrix_finish()", "task two_hop_comm_finish()"),
        ):
            self.assertEqual(
                actual[actual.index(start) : actual.index(end)],
                verified[verified.index(start) : verified.index(end)],
            )
        self.assertIn("right_matrix_dsd, 1, f16", actual)
        self.assertIn("dummy[i*seq_len_p_pe]", actual)
        with self.assertRaises(AssertionError):
            align_column_major_value(actual)


if __name__ == "__main__":
    unittest.main()
