"""Historical runs stay inspectable after authoring metadata evolves."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, json, sys, unittest
from pathlib import Path
from unittest.mock import patch

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
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
