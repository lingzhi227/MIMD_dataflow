"""Compile a frozen source probe into a separate output; never starts a simulator."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, hashlib, json, os, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("probe", type=Path)
p.add_argument("--hls", action="store_true")
a = p.parse_args()
root = a.probe.resolve()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if a.hls:
    import sys, importlib

    sys.path.insert(0, str(root / "implementation"))
    from integrity import verify_bundle, verify_codegen

    verify_bundle(root)
    verify_codegen(root)
    schedule = json.loads((root / "schedule.json").read_text())
    assert schedule["profile"] in ("mesh_mlp.v1", "mesh_projection_residual_rms.v1")
    parameters = importlib.import_module(
        schedule["profile"].split(".")[0] + "_sdk"
    ).parameters
    command = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        f"--fabric-dims={schedule['cols']+7},{schedule['rows']+2}",
        "--fabric-offsets=4,1",
        parameters(schedule),
        "-o=out",
        "--memcpy",
        "--channels=1",
    ]
    provenance_path = root / "manifest.json"
else:
    provenance_path = root / "provenance.json"
    prov = json.loads(provenance_path.read_text())
    for name, digest in prov["files"].items():
        assert sha(root / name) == digest, name
    command = json.loads((root / "sdk-command.json").read_text())
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
out = root / ("compile-check-" + stamp)
out.mkdir()
cmd = command
assert sum(v.startswith("-o=") for v in cmd) == 1
cmd = [("-o=" + out.name) if v.startswith("-o=") else v for v in cmd]
with (out / "compiler.log").open("w") as log:
    r = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
report = dict(
    compiler_accepted=r.returncode == 0,
    returncode=r.returncode,
    command=cmd,
    new_sdk_simulation=False,
    source_provenance_sha256=sha(provenance_path),
    hls_bundle=a.hls,
    driver_sha256=sha(Path(__file__)),
    compiler_log_sha256=sha(out / "compiler.log"),
    scope="Compiler-only acceptance into separate output. No execution, numerical, protocol, or performance qualification.",
)
(out / "compilation.json").write_text(json.dumps(report, indent=2) + "\n")
print(out, report, flush=True)
raise SystemExit(r.returncode)
