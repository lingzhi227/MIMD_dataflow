"""Historical runs stay inspectable after authoring metadata evolves."""

import copy, json, sys, unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from frozen_inspection import completed


class FrozenInspection(unittest.TestCase):
    def test_historical_frozen_audit_and_corruption(self):
        root = (
            ROOT
            / "tests/fixtures/history/run-20260907T090013451671Z"
        )
        results = json.loads((root / "results.json").read_text())
        with patch(
            "mesh_feed_forward.plan",
            side_effect=AssertionError("must use frozen planner"),
        ):
            report = completed(root, results)
        self.assertTrue(report["passed"])
        self.assertEqual(report["epochs"], 8)
        bad = copy.deepcopy(results)
        bad["diagnostics"][6]["rms_progress"][0][0][1] = 1
        with self.assertRaisesRegex(
            ValueError, "frozen completed-call inspection failed"
        ):
            completed(root, bad)
        self.assertEqual(results, json.loads((root / "results.json").read_text()))
