"""Keep final-only checks distinct from complete original-input qualification."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT))
from fixtures import batches, check_application
from run_profiles import numerical_summary

BUNDLE = (
    ROOT
    / "tests/fixtures/history/run-20260907T144456167691Z"
)


class InputAttentionReporting(unittest.TestCase):
    def test_branch_and_local_rounding_checks_cannot_be_omitted(self):
        rows = json.loads((BUNDLE / "application-gate.json").read_text())["checks"]
        case = dict(
            native_application_checks=rows,
            device_application_checks=rows,
            audit=dict(passed=True),
        )
        self.assertTrue(numerical_summary(case)["fixed_accuracy_passed"])
        for branch in (
            "input_normalized",
            "q_raw",
            "k_raw",
            "v_raw",
            "q",
            "k",
            "v",
            "score",
            "probability",
            "attention",
            "projection",
            "mlp_delta",
        ):
            damaged = copy.deepcopy(case)
            del damaged["device_application_checks"][0][branch]
            with self.assertRaises(ValueError):
                numerical_summary(damaged)
        damaged = copy.deepcopy(case)
        damaged["device_application_checks"][0]["q"][
            "local_pair_rounding_passed"
        ] = False
        with self.assertRaises(ValueError):
            numerical_summary(damaged)

    def test_catalog_fixture_matches_immutable_inputs(self):
        actual = batches("input_attention_mixed:64:64:256:8", 8)
        self.assertEqual(actual, json.loads((BUNDLE / "batches.json").read_text()))
        result = json.loads((BUNDLE / "reference.json").read_text())["outputs"][0]
        check = check_application(
            "input_attention_mixed:64:64:256:8", actual[0], result
        )
        self.assertTrue(check["fixed_accuracy_passed"])
        self.assertEqual(check["contract"], "shared-input-attention-final-only-v1")
        self.assertNotIn("probability", check)
