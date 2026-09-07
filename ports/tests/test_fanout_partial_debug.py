"""Incomplete device execution must not become a full qualification."""

import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from frontend import Error
from mesh_normalized_fanout_sdk import audit_cases
from normalized_fanout_debug import inspect


class PartialDebug(unittest.TestCase):
    def test_partial_records_rejected_by_full_audit_before_numerics(self):
        r = dict(
            success=False,
            runtime_instances=1,
            cases=[{}],
            diagnostics=[{}],
            launches=["hls_main"],
        )
        with self.assertRaisesRegex(Error, "full lifecycle"):
            audit_cases({}, dict(epochs=6), [{}] * 6, r)
        r["success"] = True
        with self.assertRaisesRegex(Error, "completed prefix"):
            audit_cases({}, dict(epochs=6), [{}] * 6, r, require_complete=False)

    def test_empty_prefix_cannot_pass(self):
        r = dict(
            success=False, runtime_instances=1, cases=[], diagnostics=[], launches=[]
        )
        with self.assertRaisesRegex(Error, "completed prefix"):
            audit_cases({}, dict(epochs=6), [{}] * 6, r, require_complete=False)

    def test_future_epoch_is_unavailable_not_an_index_error(self):
        s = dict(P=4, Mt=2, Nt=2, epochs=6, projections=3, instrumentation="sampled")
        v = inspect(s, dict(diagnostics=[]), "p0_0", 4, 0)
        self.assertFalse(v["available"])
        self.assertFalse(v["observed"])
        self.assertIsNone(v["half_bits"])
