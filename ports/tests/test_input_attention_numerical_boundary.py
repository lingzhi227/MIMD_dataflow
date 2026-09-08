"""Do not erase the cancellation case or mistake native precision for SDK proof."""

import json, sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from native_transport import parse_outputs
from input_attention_fixtures import check


class InputAttentionNumericalBoundary(unittest.TestCase):
    def test_actual_native_precision_boundary(self):
        root = ROOT / "tests/fixtures/history/input-attention-codegen-20260907T130338205990Z"
        b = json.loads((root / "logical-inputs.json").read_text())[3]
        study = ROOT / "tests/fixtures/history/input-attention-mixed-precision-20260907T131046512625Z"
        for name, passing in (
            ("baseline", False),
            ("wide-PV-O-Z", False),
            ("wide-V-PV-O-Z", True),
        ):
            row = parse_outputs((study / name / "output.txt").read_text())[3]
            obs = {k[10:]: v for k, v in row.items() if k.startswith("__observe_")}
            obs["v"] = obs["v_raw"]
            args = (64, 64, 256, 1e-6, 0.125, b, {"output": row["output"]}, obs)
            if passing:
                self.assertTrue(check(*args)["fixed_accuracy_passed"])
            else:
                with self.assertRaises(AssertionError):
                    check(*args)


if __name__ == "__main__":
    unittest.main()
