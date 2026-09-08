
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from sdk_process import run_sdk


class SDKWatchdog(unittest.TestCase):
    def test_fatal_is_reported_before_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = f"from pathlib import Path; import time; Path({str(root/'sim.log')!r}).write_text('FATAL: microthread ownership conflict'); time.sleep(30)"
            started = time.monotonic()
            with (root / "sdk.log").open("w") as log:
                with self.assertRaisesRegex(
                    RuntimeError, "microthread ownership conflict"
                ):
                    run_sdk(
                        [sys.executable, "-c", code], root, dict(os.environ), log, 20
                    )
            self.assertLess(time.monotonic() - started, 8)

    def test_completed_call_failure_terminates_owned_process(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = f"from pathlib import Path;import time;Path({str(root/'ready')!r}).write_text('ready');time.sleep(30)"

            def checker():
                if (root / "ready").exists():
                    raise ValueError("completed branch mismatch")

            started = time.monotonic()
            with (root / "sdk.log").open("w") as log:
                with self.assertRaisesRegex(ValueError, "completed branch mismatch"):
                    run_sdk(
                        [sys.executable, "-c", code],
                        root,
                        dict(os.environ),
                        log,
                        20,
                        progress_check=checker,
                    )
            self.assertLess(time.monotonic() - started, 8)
            self.assertEqual(
                json.loads((root / "execution-stage.json").read_text())["state"],
                "sdk_failed",
            )

    def test_timeout_terminates_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "sdk.log").open("w") as log:
                with self.assertRaises(TimeoutError):
                    run_sdk(
                        [sys.executable, "-c", "import time; time.sleep(30)"],
                        root,
                        dict(os.environ),
                        log,
                        0.2,
                    )


if __name__ == "__main__":
    unittest.main()
