"""Negative cslc probe: a library callback must not steal an SDK task ID."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = repository_root(__file__)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(bundle):
    sys.path.insert(0, str(bundle / "implementation"))
    from integrity import verify_bundle, verify_codegen

    verify_bundle(bundle)
    verify_codegen(bundle)
    root = (
        ROOT
        / "validation/evidence"
        / (
            "composed-task-collision-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    for p in bundle.glob("*.csl"):
        shutil.copy2(p, root / p.name)
    p = root / "sdk_axis_mean.csl"
    code = p.read_text()
    assert code.count("@get_local_task_id(12)") == 1
    p.write_text(code.replace("@get_local_task_id(12)", "@get_local_task_id(14)"))
    shutil.copy2(__file__, root / "driver.py")
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                parent_manifest_sha256=sha(bundle / "manifest.json"),
                change="Mean callback task12 ->14, deliberately colliding with SDK X FSM entrypoint; all other CSL bytes unchanged.",
                files={p.name: sha(p) for p in root.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT), flush=True)


def run(root):
    root = root.resolve()
    assert not (root / "report.json").exists()
    provenance = json.loads((root / "provenance.json").read_text())
    for n, h in provenance["files"].items():
        assert sha(root / n) == h
    image = Path(
        "/home/lingzhi/cerebras/sdk/2.10.1/sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif"
    )
    digest = sha(image)
    assert digest == "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"
    command = [
        "/usr/local/bin/singularity",
        "exec",
        str(image),
        "cslc",
        "layout.csl",
        "--arch=wse3",
        "--fabric-dims=23,18",
        "--fabric-offsets=4,1",
        "-o=out",
        "--memcpy",
        "--channels=1",
    ]
    (root / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(
        command, cwd=root, text=True, capture_output=True, timeout=240
    )
    (root / "compiler.log").write_text(result.stdout + result.stderr)
    report = dict(
        returncode=result.returncode,
        expected_rejection=result.returncode != 0,
        sdk_sha256=digest,
        compiler_log_sha256=sha(root / "compiler.log"),
        provenance_sha256=sha(root / "provenance.json"),
        command_sha256=sha(root / "command.json"),
        scope="Compile-only negative task-collision experiment. Diagnostic text must be inspected to establish rejection reason; no simulator executed.",
    )
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", type=Path)
    p.add_argument("--execute", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.prepare.resolve())
    else:
        run(a.execute)
