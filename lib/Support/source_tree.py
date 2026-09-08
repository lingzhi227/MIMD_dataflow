"""Resolve authoring files by their stable frozen-bundle logical names.

The checkout is organized by compiler responsibility; immutable SDK bundles keep
flat modules plus include/ and runtime/. No shadow source tree or symlinks exist.
"""
from pathlib import Path
import json
from functools import lru_cache

BASE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def checkout():
    # A frozen snapshot must never accidentally import live authoring assets.
    if (BASE / "frontend.py").is_file():
        return None
    for p in BASE.parents:
        marker = p / "hls-layout.json"
        if marker.is_file():
            return p, json.loads(marker.read_text())["compiler_sources"]
    raise RuntimeError("Missing compiler layout or frozen implementation")


def source_path(name):
    name = str(name)
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError("Unsafe compiler source path")
    current = checkout()
    if current is None:
        return BASE / relative
    root, mapping = current
    if name == 'runtime':
        return root / 'runtime/csl'
    if name == 'include':
        return root / 'include/pragma'
    return root / mapping[name]


def logical_name(path):
    path = Path(path).resolve()
    current = checkout()
    if current is None:
        return str(path.relative_to(BASE))
    root, mapping = current
    reverse = {str((root / p).resolve()): name for name, p in mapping.items()}
    return reverse[str(path)]
