"""Prepare a one-PE compiler-only reproducer; never launches a simulator."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import datetime, json, hashlib, shutil
from pathlib import Path

ROOT = repository_root(__file__)
root = (
    ROOT
    / "validation/evidence"
    / (
        "blocked-map-compiler-minimal-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
)
root.mkdir()
old = (
    ROOT / "benchmarks/inference/waferllm/cache_attention_5x256x512_8x8/run-20260907T215451004899Z"
)
for variant in ("mapped", "indexed"):
    dest = root / variant
    dest.mkdir()
    shutil.copyfile(old / "batched_matmul_local.csl", dest / "batched_matmul_local.csl")
    source = (
        old / "batched_matmul_blocked.csl"
        if variant == "mapped"
        else ROOT / "runtime/csl/batched_matmul_blocked.csl"
    )
    shutil.copyfile(source, dest / "batched_matmul_blocked.csl")
    (dest / "pe.csl").write_text("""param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const mm=@import_module("batched_matmul_blocked.csl",.{.batches=2,.inner=4,.columns=2,.block_size=2});
var X=@zeros([8]f16);var W=@zeros([8]f16);var Y=@zeros([4]f16);
fn main() void {mm.compute(&X,&W,&Y);sys.unblock_cmd_stream();}
var xp:[*]f16=&X;var wp:[*]f16=&W;var yp:[*]f16=&Y;
comptime {@export_symbol(xp,"X");@export_symbol(wp,"W");@export_symbol(yp,"Y");@export_symbol(main);}
""")
    (dest / "layout.csl").write_text(
        """const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});
layout {@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});
@export_name("X",[*]f16,true);@export_name("W",[*]f16,true);@export_name("Y",[*]f16,true);@export_name("main",fn()void);}
"""
    )
    (dest / "command.json").write_text(
        json.dumps(
            [
                "cslc",
                "layout.csl",
                "--arch=wse3",
                "--fabric-dims=8,3",
                "--fabric-offsets=4,1",
                "-o=out",
                "--memcpy",
                "--channels=1",
            ]
        )
        + "\n"
    )
(root / "worker.py").write_text("""import subprocess,json,sys,os
from pathlib import Path
root=Path(sys.argv[1]).resolve();os.chdir(root)
r=subprocess.run(json.loads((root/'command.json').read_text()))
(root/'compiler-result.json').write_text(json.dumps(dict(returncode=r.returncode,compiled=r.returncode==0,simulator_executed=False))+'\\n')
""")
(root / "execute.py").write_text("""import subprocess,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
for name,digest in json.loads((root/'provenance.json').read_text())['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
sif=Path('/home/lingzhi/cerebras/sdk/2.10.1/sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif')
with sif.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
assert digest=='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d'
reports={}
for variant in ('mapped','indexed'):
 dest=root/variant;assert not (dest/'compiler.log').exists()
 with (dest/'compiler.log').open('w') as log:subprocess.run(['/home/lingzhi/cerebras/sdk/2.10.1/cs_python',str(root/'worker.py'),str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
 reports[variant]=json.loads((dest/'compiler-result.json').read_text())
 reports[variant]['log_sha256']=hashlib.sha256((dest/'compiler.log').read_bytes()).hexdigest()
(root/'review.json').write_text(json.dumps(dict(scope='One PE 2x4x2 block2 compiler isolation only; no simulator or numerical execution',sdk_sha256=digest,variants=reports),indent=2)+'\\n')
print(json.dumps(reports))
""")
shutil.copyfile(__file__, root / "prepare.py")
(root / "provenance.json").write_text(
    json.dumps(
        dict(
            scope="Compiler-only mapped versus indexed merge/narrow, identical contraction semantics",
            files={
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*")
                if p.is_file()
            },
        ),
        indent=2,
    )
    + "\n"
)
print(root.relative_to(ROOT))
