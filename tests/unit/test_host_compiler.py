"""Host Clang selection is explicit, preserved as one argv element and shared."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import os, sys, unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from host_compiler import executable
from frontend import host_compiler as frontend_compiler
from compile import host_compiler as native_compiler


class HostCompiler(unittest.TestCase):
    def test_default_and_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(executable(), "clang++")
        with patch.dict(
            os.environ, {"HLS_CLANGXX": "/example toolchain/bin/clang++-17"}
        ):
            self.assertEqual(frontend_compiler(), "/example toolchain/bin/clang++-17")
            self.assertEqual(frontend_compiler(), native_compiler())
        with patch.dict(os.environ, {"HLS_CLANGXX": " "}):
            with self.assertRaises(ValueError):
                executable()
