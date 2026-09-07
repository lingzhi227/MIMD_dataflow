"""Exhaustive positive finite IEEE-half sqrt/reciprocal and expression-order probe."""

import argparse, datetime, json, shutil
from pathlib import Path
from probe_runtime import sha, execute, half_vector_worker

ROOT = Path(__file__).resolve().parents[1]
PORTS = [
    "input",
    "sqrt_value",
    "reciprocal",
    "library_inv",
    "mean",
    "argument",
    "chain_sqrt",
    "chain_inverse",
    "expression_chain",
    "expr_fma",
    "direct_fma",
]


def prepare():
    import numpy as np

    root = (
        ROOT
        / "evidence"
        / (
            "rms-math-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    pe = 'param N:i16;param memcpy_params;\nconst sys=@import_module("<memcpy/memcpy>",memcpy_params);\nconst math=@import_module("<math>");\n'
    pe += "".join(
        f"var {name}=@zeros([N]f16);var ptr_{name}:[*]f16=&{name};\n" for name in PORTS
    )
    pe += """var progress=@zeros([1]u16);var ptr_progress:[*]u16=&progress;
var normal_buffer=@zeros([1]f16);var negative_buffer=@zeros([1]f16);
const normal_dsd=@get_dsd(mem1d_dsd,.{.base_address=&normal_buffer,.extent=1});
const negative_dsd=@get_dsd(mem1d_dsd,.{.base_address=&negative_buffer,.extent=1});
const direct_dsd=@get_dsd(mem1d_dsd,.{.base_address=&direct_fma,.extent=1});
fn main() void {
 for(@range(i16,N)) |i| {
  sqrt_value[i]=math.sqrt_f16(input[i]);
  reciprocal[i]=1.0/sqrt_value[i];
  library_inv[i]=math.inv_f16(sqrt_value[i]);
  mean[i]=input[i]/64.0;
  argument[i]=mean[i]+@as(f16,0.000001);
  chain_sqrt[i]=math.sqrt_f16(argument[i]);
  chain_inverse[i]=1.0/chain_sqrt[i];
  var sum:f16=input[i];sum=sum/64.0;
  expression_chain[i]=1.0/math.sqrt_f16(sum+@as(f16,0.000001));
  const normal=@bitcast(f16,(@bitcast(u16,input[i])&@as(u16,0x03ff))|@as(u16,0x3c00));
  expr_fma[i]= -normal + normal*normal;
  normal_buffer[0]=normal;negative_buffer[0]=-normal;
  @fmach(@increment_dsd_offset(direct_dsd,i,f16),negative_dsd,normal_dsd,normal);
 }
 progress[0]+=1;sys.unblock_cmd_stream();
}
comptime {
"""
    pe += "".join(f' @export_symbol(ptr_{name},"{name}");\n' for name in PORTS)
    pe += ' @export_symbol(ptr_progress,"progress");@export_symbol(main);\n}\n'
    (root / "pe.csl").write_text(pe)
    layout = """param N:u16;
const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});
layout {@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.N=@as(i16,N),.memcpy_params=memcpy.get_params(0)});
"""
    layout += "".join(f' @export_name("{name}",[*]f16,true);\n' for name in PORTS)
    layout += (
        ' @export_name("progress",[*]u16,true);@export_name("main",fn()void);\n}\n'
    )
    (root / "layout.csl").write_text(layout)
    x = np.arange(0x7C00, dtype=np.uint16).view(np.float16).reshape(-1, 512)
    (root / "inputs.json").write_text(json.dumps(x.tolist()) + "\n")
    (root / "schema.json").write_text(
        json.dumps(dict(input="input", outputs=PORTS[1:], length=512)) + "\n"
    )
    (root / "sdk-command.json").write_text(
        json.dumps(
            [
                "cslc",
                "layout.csl",
                "--arch=wse3",
                "--fabric-dims=8,3",
                "--fabric-offsets=4,1",
                "--params=N:512",
                "-o=out",
                "--memcpy",
                "--channels=1",
            ]
        )
        + "\n"
    )
    shutil.copyfile(__file__, root / "driver.py")
    shutil.copyfile(ROOT / "experiments/probe_runtime.py", root / "probe_runtime.py")
    shutil.copyfile(ROOT / "toolchain/sdk_process.py", root / "sdk_process.py")
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                prepared_only=True,
                domain="All31744nonnegative finite IEEE-half encodings, including positive zero.62changed calls of512values.",
                sdk_math_sources=__import__("json").loads(
                    (ROOT / "references/sdk-math-2.10.1/PROVENANCE.json").read_text()
                ),
                files={p.name: sha(p) for p in root.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--execute", type=Path)
    g.add_argument("--worker", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.execute:
        execute(a.execute.resolve(), timeout=1800)
    else:
        half_vector_worker(a.worker.resolve())
