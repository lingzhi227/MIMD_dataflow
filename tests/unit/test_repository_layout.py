"""Directory contracts must preserve public entry points and frozen isolation."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from bootstrap import configure
configure(ROOT)
from source_tree import source_path, logical_name


class RepositoryLayout(unittest.TestCase):
    def test_compiler_manifest_is_complete_and_names_are_unique(self):
        layout = json.loads((ROOT / 'hls-layout.json').read_text())
        modules = []
        for name, relative in layout['compiler_sources'].items():
            self.assertTrue((ROOT / relative).is_file(), relative)
            self.assertEqual(source_path(name), ROOT / relative)
            self.assertEqual(logical_name(ROOT / relative), name)
            if name.endswith('.py'):
                modules.append(Path(name).stem)
        self.assertEqual(len(modules), len(set(modules)))

    def test_every_profile_resolves_to_source_and_contract(self):
        layout = json.loads((ROOT / 'hls-layout.json').read_text())
        for key, relative in layout['profiles'].items():
            self.assertTrue((ROOT / relative / 'hls.cpp').is_file(), key)
            self.assertTrue((ROOT / relative / 'PORT.json').is_file(), key)
        for profile in json.loads((ROOT / 'benchmarks/catalog.json').read_text()):
            self.assertIn(profile['project'] + '/' + profile['kernel'], layout['profiles'])

    def test_frozen_lookup_never_reads_authoring_assets(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            shutil.copyfile(source_path('source_tree.py'), p / 'source_tree.py')
            (p / 'frontend.py').write_text('# frozen marker\n')
            (p / 'runtime').mkdir()
            (p / 'runtime/sentinel.csl').write_text('frozen-only')
            code = "import sys;sys.path.insert(0,sys.argv[1]);from source_tree import source_path;print(source_path('runtime/sentinel.csl').read_text())"
            text = subprocess.check_output([sys.executable, '-c', code, str(p)], text=True, cwd=ROOT)
            self.assertEqual(text.strip(), 'frozen-only')

    def test_unsafe_asset_path_is_rejected(self):
        for name in ['../frontend.py', '/tmp/frontend.py']:
            with self.assertRaises(ValueError):
                source_path(name)

    def test_independent_reference_is_standalone(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            for name in ('composed_ffn_fixtures.py', 'projected_cache_fixtures.py', 'cache_attention_fixtures.py'):
                shutil.copyfile(ROOT / 'tests/support' / name, p / name)
            code = "import composed_ffn_fixtures;print(callable(composed_ffn_fixtures.check))"
            result = subprocess.check_output([sys.executable, '-c', code], cwd=p, text=True)
            self.assertEqual(result.strip(), 'True')

    def test_public_compile_entry_works_outside_checkout(self):
        with tempfile.TemporaryDirectory() as td:
            output = subprocess.check_output([sys.executable, str(ROOT / 'tools/hls_compile.py'), '--help'], cwd=td, text=True)
            self.assertIn('--partitions', output)


if __name__ == '__main__':
    unittest.main()
