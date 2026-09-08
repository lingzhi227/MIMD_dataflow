"""Configure the source checkout's explicit, collision-free module search roots.

Frozen SDK bundles remain self-contained and do not use this checkout bootstrap.
"""
from pathlib import Path
import json
import sys


def repository_root(path):
    path = Path(path).resolve()
    for parent in (path, *path.parents):
        if (parent / "hls-layout.json").is_file():
            return parent
    raise RuntimeError("Cannot locate the Pragma source checkout")


def configure(root):
    root = repository_root(root)
    paths = [root / "tools", root / "tests/support", root / "experiments", root / "scripts/authoring"]
    layout = json.loads((root / "hls-layout.json").read_text())
    paths += sorted({(root / p).parent for n, p in layout["compiler_sources"].items() if n.endswith('.py')})
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root
