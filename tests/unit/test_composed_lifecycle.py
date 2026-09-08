"""Reject invalid runtime evidence before inspecting any numerical tensors."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from frontend import Error
from projected_cache_ffn_reference import audit_cases


class ComposedLifecycle(unittest.TestCase):
    def test_wrong_launch_and_premature_success_rejected_in_partial_mode(self):
        schedule = {"attention": {"P": 16, "epochs": 8}}
        for change in (
            {"launches": ["init_task"]},
            {"launches": []},
            {"success": True},
            {"success": 1},
            {"runtime_instances": True},
            {"runtime_instances": 2},
        ):
            with self.subTest(change=change):
                result = dict(
                    success=False,
                    runtime_instances=1,
                    cases=[{}],
                    diagnostics=[{}],
                    launches=["hls_main"],
                )
                result.update(change)
                with self.assertRaises(Error):
                    audit_cases(schedule, {}, [{}] * 8, result, require_complete=False)


if __name__ == "__main__":
    unittest.main()
