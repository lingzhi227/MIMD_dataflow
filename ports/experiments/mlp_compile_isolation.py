"""Fresh compile-only reachability isolation for the generated MLP lowering."""

import argparse, datetime, json, shutil, subprocess, sys
from pathlib import Path
from probe_runtime import sha, verify, execute

ROOT = Path(__file__).resolve().parents[1]


def body(s, name, replacement):
    a = s.index("fn " + name + "(")
    start = s.index("{", a)
    depth = 1
    i = start + 1
    while depth:
        depth += (s[i] == "{") - (s[i] == "}")
        i += 1
    return s[: start + 1] + replacement + s[i - 1 :]


p = argparse.ArgumentParser()
p.add_argument("--prepare", action="store_true")
p.add_argument("--execute", type=Path)
p.add_argument("--worker", type=Path)
a = p.parse_args()
if a.prepare:
    d = (
        ROOT
        / "evidence"
        / (
            "mlp-compile-isolation-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    d.mkdir()
    src = ROOT / "toolchain/runtime"
    base = (src / "mlp_pe.csl").read_text()
    variants={"explicit_progress_branch":base,"conditional_subscript":base.replace("if(phase==0){progress[0]=@as(u16,shift_round);}else{progress[1]=@as(u16,shift_round);}","progress[if(phase==0) 0 else 1]=@as(u16,shift_round);")}
    for name, s in variants.items():
        q = d / name
        q.mkdir()
        (q / "pe.csl").write_text(s)
        shutil.copyfile(src / "mlp_layout.csl", q / "layout.csl")
        for f in ("inference_comm.csl", "inference_routes.csl", "gated_local.csl"):
            shutil.copyfile(src / f, q / f)
    shutil.copyfile(__file__, d / "driver.py")
    shutil.copyfile(ROOT / "experiments/probe_runtime.py", d / "probe_runtime.py")
    shutil.copyfile(ROOT / "toolchain/sdk_process.py", d / "sdk_process.py")
    (d / "variants.json").write_text(json.dumps(list(variants)))
    (d / "provenance.json").write_text(
        json.dumps(
            dict(
                files={
                    str(p.relative_to(d)): sha(p) for p in d.rglob("*") if p.is_file()
                },
                scope="compile-only function reachability variants, not executable algorithms",
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
    for name in json.loads((d / "variants.json").read_text()):
        q = d / name
        cmd = [
            "cslc",
            "layout.csl",
            "--arch=wse3",
            "--fabric-dims=15,10",
            "--fabric-offsets=4,1",
            "--params=P:8,dim_p_pe:8,pes_p_head:8,pes_p_kv_head:8,head_dim_p_pe:8,seq_len_p_pe:8,ffn_dim_p_pe:32,sampled:1",
            "-o=out",
            "--memcpy",
            "--channels=1",
        ]
        with (q / "compile.log").open("w") as log:
            r = subprocess.run(cmd, cwd=q, stdout=log, stderr=subprocess.STDOUT)
        rows.append(
            dict(
                variant=name, returncode=r.returncode, log_sha256=sha(q / "compile.log")
            )
        )
        print(name, r.returncode, flush=True)
    (d / "results.json").write_text(
        json.dumps(
            dict(
                success=True,
                cases=rows,
                scope="probe completed, individual compiler success is returncode zero",
            ),
            indent=2,
        )
        + "\n"
    )
