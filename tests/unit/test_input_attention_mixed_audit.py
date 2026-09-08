"""Fault injection against actual mixed SDK observations, not synthetic success rows."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, json, sys, unittest
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from mesh_input_attention_mixed import verify, plan
from mesh_feed_forward_sdk import decode
from input_attention_mixed_audit import audit_cases
from frontend import Error


class MixedAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = json.loads(
            (
                ROOT
                / "tests/fixtures/history/input-attention-mixed-frontend-20260907T134515612938Z/frontend.json"
            ).read_text()
        )
        raw["instrumentation"] = "counters"
        cls.m = verify(raw, 1, 2)
        cls.s = plan(cls.m)
        root = ROOT / "tests/fixtures/history/input-attention-codegen-20260907T133613178311Z"
        cls.bs = json.loads((root / "logical-inputs.json").read_text())[:1]
        row = json.loads((root / "results.json").read_text())["cases"][0]
        cls.r = dict(
            success=True,
            runtime_instances=1,
            launches=["hls_main"],
            diagnostics=[row],
            cases=[decode(cls.s, cls.m, row)],
        )

    def test_actual_sdk_case_passes(self):
        self.assertTrue(audit_cases(self.s, self.m, self.bs, self.r)["passed"])

    def test_corrupted_stages_protocol_and_word_width_reject(self):
        ports = [
            "residual",
            "gamma",
            "q_weight",
            "input_normalized",
            "input_q_raw",
            "input_k_raw",
            "x",
            "attention_k",
            "attention_logits",
            "mixed_v",
            "mixed_probability_snapshot",
            "mixed_a",
            "mixed_projection",
            "mixed_z",
            "mixed_normalized",
            "normalized",
            "down_snapshot",
            "result",
            "up_accumulator",
            "gate_accumulator",
            "wide_accumulator",
            "input_prefix_progress",
            "prelude_progress",
            "attention_progress",
            "score_progress",
            "score_roots",
            "rms_progress",
            "attention_softmax_progress",
            "progress",
            "queues",
        ]
        from mesh_input_attention_mixed_sdk import WIDE_PORTS

        for port in ports:
            with self.subTest(port=port):
                r = copy.deepcopy(self.r)
                old = r["diagnostics"][0][port][0][0][0]
                r["diagnostics"][0][port][0][0][0] = old ^ (
                    0x00800000
                    if port in WIDE_PORTS
                    else 0x0008 if port == "queues" else 0x0100
                )
                with self.assertRaises((AssertionError, Error)):
                    audit_cases(self.s, self.m, self.bs, r)
        r = copy.deepcopy(self.r)
        r["diagnostics"][0]["mixed_v"][0][0][0] = 2**32
        with self.assertRaisesRegex(Error, "raw word extent/range"):
            audit_cases(self.s, self.m, self.bs, r)

    def test_incomplete_lifecycle_and_decoded_output_reject(self):
        r = copy.deepcopy(self.r)
        r["runtime_instances"] = 2
        with self.assertRaisesRegex(Error, "lifecycle"):
            audit_cases(self.s, self.m, self.bs, r)
        r = copy.deepcopy(self.r)
        r["cases"][0]["output"][0] += 1
        with self.assertRaises(AssertionError):
            audit_cases(self.s, self.m, self.bs, r)
        r = copy.deepcopy(self.r)
        r["success"] = False
        with self.assertRaisesRegex(Error, "complete mixed"):
            audit_cases(self.s, self.m, self.bs, r)
        self.assertFalse(
            audit_cases(self.s, self.m, self.bs, r, require_complete=False)["complete"]
        )


if __name__ == "__main__":
    unittest.main()
