"""An internally consistent manifest must also carry the generator's dependencies."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import hashlib, json, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from compile import build
from attention_fixtures import batches


class FrozenCodegen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        source = (
            (ROOT / "benchmarks/inference/waferllm/attention_64x128_8x8/hls.cpp")
            .read_text()
            .replace("<64,128,", "<8,16,")
            .replace("rows=8 cols=8", "rows=4 cols=4")
        )
        p = cls.root / "hls.cpp"
        p.write_text(source)
        cls.bundle = build(
            p,
            cls.root / "good",
            epochs=1,
            bound=1,
            batches=batches(8, 16)[:1],
            instrumentation="counters",
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def check_snapshot(self, p):
        return subprocess.run(
            [
                sys.executable,
                "-c",
                'import sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from integrity import verify_codegen;verify_codegen(p)',
                str(p),
            ],
            capture_output=True,
            text=True,
        )

    def test_independent_codegen(self):
        self.assertEqual(self.check_snapshot(self.bundle).returncode, 0)

    def test_missing_unlisted_dependency_fails_before_sdk(self):
        p = self.root / "missing"
        shutil.copytree(self.bundle, p)
        manifest = json.loads((p / "manifest.json").read_text())
        name = "runtime/attention_pe.csl"
        del manifest["implementation"][name]
        (p / "implementation" / name).unlink()
        (p / "manifest.json").write_text(json.dumps(manifest))
        r = self.check_snapshot(p)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("attention_pe.csl", r.stderr)

    def test_coherently_rehashed_target_disagrees_with_codegen(self):
        p = self.root / "diverged"
        shutil.copytree(self.bundle, p)
        (p / "pe.csl").write_text(
            (p / "pe.csl").read_text() + "\n// injected target change\n"
        )
        manifest = json.loads((p / "manifest.json").read_text())
        manifest["files"]["pe.csl"] = hashlib.sha256(
            (p / "pe.csl").read_bytes()
        ).hexdigest()
        (p / "manifest.json").write_text(json.dumps(manifest))
        r = self.check_snapshot(p)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("frozen codegen mismatch", r.stderr)


if __name__ == "__main__":
    unittest.main()
