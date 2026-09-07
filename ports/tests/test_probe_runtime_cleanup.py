"""A failed SDK initialization must release the probe runtime, not leak it."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "probe_cleanup_subject", ROOT / "experiments/probe_runtime.py"
)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class ProbeCleanup(unittest.TestCase):
    def test_failed_initialization_stops_runtime_without_success(self):
        runner = Mock()
        runner.launch.side_effect = RuntimeError("initialization failed")
        sdk = ModuleType("cerebras.sdk.runtime.sdkruntimepybind")
        sdk.SdkRuntime = Mock(return_value=runner)
        sdk.MemcpyDataType = Mock()
        sdk.MemcpyOrder = Mock()
        sdk.SimfabConfig = Mock()
        sdk.SdkTarget = Mock()
        sdk.get_platform = Mock()
        utils = ModuleType("cerebras.sdk.sdk_utils")
        utils.input_array_to_u32 = Mock()
        modules = {
            name: ModuleType(name)
            for name in ["cerebras", "cerebras.sdk", "cerebras.sdk.runtime"]
        }
        modules.update(
            {
                "cerebras.sdk.runtime.sdkruntimepybind": sdk,
                "cerebras.sdk.sdk_utils": utils,
            }
        )
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, data in [
                ("provenance.json", {"files": {}}),
                (
                    "schema.json",
                    dict(
                        rows=1,
                        cols=1,
                        inputs={"x": 1},
                        outputs={"x": 1},
                        immutable=["x"],
                        initialize="initialize",
                        launch="compute",
                    ),
                ),
                ("sdk-command.json", ["cslc"]),
                ("runtime-options.json", {}),
            ]:
                (root / name).write_text(json.dumps(data))
            try:
                with patch.dict(sys.modules, modules), patch("subprocess.run"):
                    with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                        subject.mesh_half_worker(root)
                runner.stop.assert_called_once_with()
                self.assertFalse((root / "results.json").exists())
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
