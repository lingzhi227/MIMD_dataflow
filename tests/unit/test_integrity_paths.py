"""Nested diagnostic evidence remains confined and content verified."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from integrity import verify_bundle


class IntegrityPaths(unittest.TestCase):
    def test_nested_evidence_and_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "diagnostic").mkdir()
            p = root / "diagnostic/output.txt"
            p.write_text("observed")
            manifest = dict(
                files={
                    "diagnostic/output.txt": hashlib.sha256(p.read_bytes()).hexdigest()
                }
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verify_bundle(root, False), manifest)
            p.write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_bundle(root, False)

    def test_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "bundle"
            root.mkdir()
            outside = base / "outside"
            outside.write_text("observed")
            (root / "link").symlink_to(outside)
            for name in (
                "../outside",
                str(outside),
                "link",
                "a/../../outside",
                "a//b",
                ".",
            ):
                (root / "manifest.json").write_text(
                    json.dumps(
                        dict(
                            files={
                                name: hashlib.sha256(outside.read_bytes()).hexdigest()
                            }
                        )
                    )
                )
                with self.assertRaisesRegex(ValueError, "unsafe artifact path"):
                    verify_bundle(root, False)
