"""SDK entry point must reject failed or stale native math evidence before execution."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ApplicationGate(unittest.TestCase):
    def test_failed_or_stale_math_gate_precedes_sdk(self):
        for passed, stale in [(False, False), (True, True)]:
            with self.subTest(
                passed=passed, stale=stale
            ), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                impl = root / "implementation"
                impl.mkdir()
                # Integrity is isolated here: this test concerns ordering after
                # integrity validation, which has its own mutation regressions.
                (impl / "integrity.py").write_text(
                    'def verify_bundle(root):\n return {"application_gate":"application-gate.json", "files":{"application-gate.json":"validated"}, "epochs":1}\n'
                    'def verify_codegen(root):\n raise RuntimeError("PREFLIGHT_REACHED")\n'
                )
                (impl / "sdk_process.py").write_text(
                    'def run_sdk(*args):\n raise RuntimeError("SDK_REACHED")\n'
                )
                (impl / "validate.py").write_text("def audit(root):\n return {}\n")
                (root / "native-output.txt").write_text("native\n")
                (root / "application-reference.py").write_text("reference\n")
                digest = lambda name: hashlib.sha256(
                    (root / name).read_bytes()
                ).hexdigest()
                gate = dict(
                    passed=passed,
                    checks=[{}],
                    native_output_sha256=(
                        "stale" if stale else digest("native-output.txt")
                    ),
                    reference_sha256=digest("application-reference.py"),
                )
                (root / "application-gate.json").write_text(json.dumps(gate))
                r = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "experiments/execute_frozen_bundle.py"),
                        str(root),
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("AssertionError", r.stderr)
                self.assertNotIn("PREFLIGHT_REACHED", r.stderr)
                self.assertNotIn("SDK_REACHED", r.stderr)
                self.assertFalse((root / "sdk.log").exists())


if __name__ == "__main__":
    unittest.main()
