"""Compile-only minimal SDK diagnostic: conditional array subscript versus typed index/branch."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, shutil, subprocess, sys, os
from pathlib import Path
from probe_runtime import sha, execute, verify

ROOT = repository_root(__file__)
p = argparse.ArgumentParser()
p.add_argument("--prepare", action="store_true")
p.add_argument("--execute", type=Path)
p.add_argument("--worker", type=Path)
a = p.parse_args()
if a.prepare:
    d = (
        ROOT
        / "validation/evidence"
        / (
            "conditional-subscript-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    d.mkdir()
    variants = {
        "inline": "values[if(phase==0) 0 else 1]=1;",
        "typed": "const index:i16=if(phase==0) 0 else 1;values[index]=1;",
        "branch": "if(phase==0){values[0]=1;}else{values[1]=1;}",
    }
    for name, expr in variants.items():
        q = d / name
        q.mkdir()
        (q / "layout.csl").write_text(
            'layout { @set_rectangle(1,1); @set_tile_code(0,0,"pe.csl",.{}); @export_name("run",fn()void); }\n'
        )
        (q / "pe.csl").write_text(
            "var values=@zeros([2]u16);var phase:i16=0; fn run() void {"
            + expr
            + "phase+=1;} comptime {@export_symbol(run); }\n"
        )
    shutil.copyfile(__file__, d / "driver.py")
    shutil.copyfile(ROOT / "experiments/probe_runtime.py", d / "probe_runtime.py")
    shutil.copyfile(ROOT / "lib/Runtime/sdk_process.py", d / "sdk_process.py")
    (d / "provenance.json").write_text(
        json.dumps(
            dict(
                files={
                    str(p.relative_to(d)): sha(p) for p in d.rglob("*") if p.is_file()
                },
                scope="minimal compile-only conditional array subscript; no application qualification",
            ),
            indent=2,
        )
        + "\n"
    )
    print(d.relative_to(ROOT))
elif a.execute:
    execute(a.execute.resolve())
elif a.worker:
    d = a.worker.resolve()
    verify(d)
    rows = []
    for name in ("inline", "typed", "branch"):
        q = d / name
        with (q / "compile.log").open("w") as log:
            r = subprocess.run(
                [
                    "cslc",
                    "layout.csl",
                    "--arch=wse3",
                    "--fabric-dims=8,3",
                    "--fabric-offsets=4,1",
                    "--memcpy",
                    "--channels=1",
                    "-o=out",
                ],
                cwd=q,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        rows.append(
            dict(
                variant=name, returncode=r.returncode, log_sha256=sha(q / "compile.log")
            )
        )
    (d / "results.json").write_text(
        json.dumps(
            dict(
                success=True,
                cases=rows,
                scope="diagnostic compile return codes; success means probe completed",
            ),
            indent=2,
        )
        + "\n"
    )
