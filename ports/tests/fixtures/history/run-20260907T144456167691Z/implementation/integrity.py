"""Fail closed if deployed sources, IR or implementation differ from build."""

import hashlib, json
from pathlib import Path


def artifact_path(root, name):
    """Allow nested evidence while rejecting traversal and escaping symlinks."""
    base = Path(root).resolve()
    relative = Path(name)
    if (
        not name
        or relative.is_absolute()
        or ".." in relative.parts
        or str(relative) != name
    ):
        raise ValueError("unsafe artifact path")
    path = (base / relative).resolve()
    if base not in path.parents:
        raise ValueError("unsafe artifact path")
    return path


def verify_bundle(root, implementation=True):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = artifact_path(root, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("artifact hash mismatch: " + name)
    if implementation:
        base = Path(__file__).resolve().parent
        for name, digest in manifest["implementation"].items():
            path = artifact_path(base, name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("implementation hash mismatch: " + name)
    if manifest.get("implementation_snapshot"):
        for name, digest in manifest["implementation"].items():
            if (
                hashlib.sha256(
                    artifact_path(root / "implementation", name).read_bytes()
                ).hexdigest()
                != digest
            ):
                raise ValueError("implementation snapshot mismatch: " + name)
    return manifest


def verify_codegen(root):
    """Check that the frozen backend independently regenerates every target file."""
    import tempfile
    from backend import generate

    root = Path(root)
    manifest = verify_bundle(root)
    schedule = json.loads((root / "schedule.json").read_text())
    with tempfile.TemporaryDirectory() as td:
        generated = Path(td)
        generate(schedule, generated)
        files = sorted(p for p in generated.rglob("*") if p.is_file())
        if not files:
            raise ValueError("frozen codegen produced no files")
        for path in files:
            name = str(path.relative_to(generated))
            if (
                name not in manifest["files"]
                or path.read_bytes() != (root / name).read_bytes()
            ):
                raise ValueError("frozen codegen mismatch: " + name)
    return dict(
        passed=True, generated_files=[str(p.relative_to(generated)) for p in files]
    )
