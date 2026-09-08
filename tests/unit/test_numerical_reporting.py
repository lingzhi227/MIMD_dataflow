
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

import sys
import unittest
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT))
from run_profiles import numerical_summary


class NumericalReporting(unittest.TestCase):
    def test_resident_probability_contract_includes_mass_and_nonnegativity(self):
        value = {
            "contract": "score-softmax-half-normwise-v1",
            "fixed_accuracy_passed": True,
        }
        summary = numerical_summary({"native_application_checks": [value]})
        self.assertEqual(summary["max_row_mass_error_limit"], 0.01)
        self.assertTrue(summary["nonnegative_probability_required"])
        self.assertFalse(summary["zero_reference_requires_exact_zero"])
        self.assertEqual(summary["relative_l2_limit"], 0.015)

    def test_fft_normwise_contract_has_no_componentwise_tolerance(self):
        value = {"contract": "fft-f32-normwise-v1", "fixed_accuracy_passed": True}
        native = numerical_summary({"native_application_checks": [value]})
        self.assertIsNone(native["device_fixed_accuracy_passed"])
        self.assertNotIn("rtol", native)
        self.assertNotIn("atol", native)
        self.assertTrue(native["per_component_accuracy_not_implied"])
        self.assertEqual(native["relative_l2_limit"], 2e-5)
        self.assertEqual(native["max_error_over_reference_peak_limit"], 3e-5)
        failed = numerical_summary(
            {
                "native_application_checks": [value],
                "device_application_checks": [value],
                "audit": {"passed": False},
            }
        )
        self.assertFalse(failed["fixed_accuracy_passed"])
        with self.assertRaisesRegex(ValueError, "mixed normwise"):
            numerical_summary(
                {
                    "native_application_checks": [value],
                    "device_application_checks": [
                        {"contract": "factor-v1", "fixed_accuracy_passed": True}
                    ],
                }
            )

    def test_rms_reports_half_normwise_limits(self):
        value = {"contract": "rms-half-normwise-v1", "fixed_accuracy_passed": True}
        summary = numerical_summary(
            {
                "native_application_checks": [value],
                "device_application_checks": [value],
                "audit": {"passed": True},
            }
        )
        self.assertTrue(summary["fixed_accuracy_passed"])
        self.assertNotIn("rtol", summary)
        self.assertEqual(summary["relative_l2_limit"], 0.01)
        self.assertEqual(summary["max_error_over_reference_peak_limit"], 0.015)

    def test_softmax_and_resident_do_not_claim_f32_tolerance(self):
        for contract, limits in [
            ("softmax-half-normwise-v1", (0.01, 0.015)),
            ("normalized-matmul-half-normwise-v1", (0.015, 0.02)),
            ("normalized-fanout-half-normwise-v1", (0.015, 0.02)),
        ]:
            v = dict(contract=contract, fixed_accuracy_passed=True)
            r = numerical_summary(
                dict(
                    native_application_checks=[v],
                    device_application_checks=[v],
                    audit=dict(passed=True),
                )
            )
            self.assertNotIn("rtol", r)
            self.assertNotIn("atol", r)
            self.assertEqual(
                (r["relative_l2_limit"], r["max_error_over_reference_peak_limit"]),
                limits,
            )
            if contract.startswith("softmax"):
                self.assertEqual(r["max_row_mass_error_limit"], 0.01)
                self.assertTrue(r["nonnegative_probability_required"])
                self.assertFalse(r["zero_reference_requires_exact_zero"])
            r = numerical_summary(
                dict(native_application_checks=[v], audit=dict(passed=False))
            )
            self.assertFalse(r["fixed_accuracy_passed"])

    def test_gating_keeps_absolute_subnormal_allowance(self):
        value = dict(contract="gated-activation-half-v1", fixed_accuracy_passed=True)
        case = dict(
            native_application_checks=[value],
            device_application_checks=[value],
            audit=dict(passed=True),
        )
        result = numerical_summary(case)
        self.assertTrue(result["fixed_accuracy_passed"])
        self.assertEqual(result["componentwise_relative_term"], 0.004)
        self.assertEqual(result["absolute_rounding_term"], "2^-24*(1+abs(up))")
        self.assertNotIn("rtol", result)
        self.assertNotIn("relative_l2_limit", result)
        case["audit"]["passed"] = False
        self.assertFalse(numerical_summary(case)["fixed_accuracy_passed"])
        case["device_application_checks"] = [
            dict(contract="rms-half-normwise-v1", fixed_accuracy_passed=True)
        ]
        with self.assertRaisesRegex(ValueError, "mixed gated"):
            numerical_summary(case)

    def test_pair_rotation_cancellation_contract(self):
        v = dict(contract="pair-rotation-half-v1", fixed_accuracy_passed=True)
        c = dict(
            native_application_checks=[v],
            device_application_checks=[v],
            audit=dict(passed=True),
        )
        r = numerical_summary(c)
        self.assertTrue(r["fixed_accuracy_passed"])
        self.assertEqual(r["componentwise_product_magnitude_term"], 0.0015)
        self.assertEqual(r["absolute_rounding_term"], 2**-23)
        self.assertNotIn("rtol", r)
        c["audit"]["passed"] = False
        self.assertFalse(numerical_summary(c)["fixed_accuracy_passed"])
        c["device_application_checks"] = [
            dict(contract="gated-activation-half-v1", fixed_accuracy_passed=True)
        ]
        with self.assertRaisesRegex(ValueError, "mixed pair"):
            numerical_summary(c)

    def test_fixed_accuracy_failure_is_not_roundoff_failure(self):
        summary = numerical_summary(
            {
                "native_application_checks": [
                    {"roundoff_passed": True, "fixed_accuracy_passed": False}
                ]
            }
        )
        self.assertTrue(summary["arithmetic_roundoff_passed"])
        self.assertFalse(summary["fixed_accuracy_passed"])
        self.assertIsNone(summary["device_fixed_accuracy_passed"])

    def test_device_partial_failure_is_not_masked(self):
        valid = {"roundoff_passed": True, "fixed_accuracy_passed": True}
        summary = numerical_summary(
            {
                "native_application_checks": [valid],
                "device_application_checks": [valid],
                "audit": {
                    "arithmetic_roundoff_passed": True,
                    "fixed_accuracy_passed": False,
                },
            }
        )
        self.assertTrue(summary["arithmetic_roundoff_passed"])
        self.assertFalse(summary["fixed_accuracy_passed"])

    def test_factor_contract_is_not_called_roundoff(self):
        value = {"contract": "factor-v1", "fixed_accuracy_passed": True}
        summary = numerical_summary(
            {
                "native_application_checks": [value],
                "device_application_checks": [value],
                "audit": {"fixed_accuracy_passed": False},
            }
        )
        self.assertFalse(summary["fixed_accuracy_passed"])
        self.assertNotIn("arithmetic_roundoff_passed", summary)

    def test_legacy_checks_are_not_relabelled(self):
        self.assertIsNone(numerical_summary({"native_application_checks": [None]}))


if __name__ == "__main__":
    unittest.main()
