"""Isolated pinned Prefill SiLU/product control, preserving both source bodies."""

import argparse, datetime, json, shutil, sys
from pathlib import Path
from probe_runtime import read, sha, verify, execute

ROOT = Path(__file__).resolve().parents[1]


def prepare(hls):
    s = read(hls / "schedule.json")
    assert s["profile"] == "mesh_swiglu.v1"
    root = (
        ROOT
        / "evidence"
        / (
            "swiglu-source-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    src = ROOT / "projects/waferllm/upstream/Prefill/src/prefill.csl"
    original = src.read_text()
    body = original[original.index("fn silu_kernel") : original.index("fn h2_matmul")]
    # Independent wrapper: only source-owned arithmetic and DSR schedule in body.
    pe = """param memcpy_params;param length:i16;param sampled:i16;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const math_lib=@import_module("<math>");const time=@import_module("<time>");
var z1=@zeros([length]f16);var z2=@zeros([length]f16);var z3=@zeros([length]f16);
const z1_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{length}->z1[i]});
const z2_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{length}->z2[i]});
const z3_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{length}->z3[i]});
const comp_dest_dsr_1=@get_dsr(dsr_dest,1);const comp_src0_dsr_1=@get_dsr(dsr_src0,1);const comp_src1_dsr_1=@get_dsr(dsr_src1,1);
var progress=@zeros([3]u16);var timing=@zeros([6]u16);var start=@zeros([3]u16);var end=@zeros([3]u16);
fn prefill_struct() void {}
fn hls_main() void {
 time.enable_tsc();time.get_timestamp(&start);progress[0]=0;progress[1]=0;
 z3_comp();progress[0]=1;progress[1]=1;progress[2]+=1;
 time.get_timestamp(&end);time.disable_tsc();
 for(@range(i16,3)) |i| {timing[i]=start[i];timing[i+3]=end[i];}
 sys.unblock_cmd_stream();
}
var uptr:[*]f16=&z1;var gptr:[*]f16=&z2;var rptr:[*]f16=&z3;var aptr:[*]f16=&z2;
var pptr:[*]u16=&progress;var tptr:[*]u16=&timing;
comptime {@export_symbol(uptr,"up");@export_symbol(gptr,"gate");@export_symbol(rptr,"result");@export_symbol(aptr,"activated");@export_symbol(pptr,"progress");@export_symbol(tptr,"timing");@export_symbol(hls_main);}
""" + body
    (root / "pe.csl").write_text(pe)
    (root / "original-kernels.csl").write_text(body)
    for n in [
        "layout.csl",
        "schedule.json",
        "semantic.json",
        "runtime-options.json",
        "WaferLLM-LICENSE.txt",
    ]:
        shutil.copyfile(hls / n, root / n)
    (root / "batches.json").write_text(
        json.dumps(read(hls / "batches.json")[:2]) + "\n"
    )
    shutil.copytree(hls / "implementation", root / "implementation")
    for n in ["probe_runtime.py"]:
        shutil.copyfile(ROOT / "experiments" / n, root / n)
    shutil.copyfile(ROOT / "toolchain/sdk_process.py", root / "sdk_process.py")
    shutil.copyfile(Path(__file__), root / "driver.py")
    files = {
        str(p.relative_to(root)): sha(p)
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                source_repository="https://github.com/MeshInfra/WaferLLM.git",
                source_sha256=sha(src),
                hls_bundle=str(hls.relative_to(ROOT)),
                files=files,
                scope="Isolated original silu_kernel and z3_comp bodies; empty prefill_struct completion callback. Original gate is overwritten in place. No full Prefill execution. Shared HLS SDK transport; independent source arithmetic. Sampled HLS copies activation; source aliases existing z2. Counter comparison avoids this copy asymmetry.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", type=Path)
    p.add_argument("--execute", type=Path)
    p.add_argument("--worker", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.prepare.resolve())
    elif a.execute:
        execute(a.execute.resolve(), 1800)
    elif a.worker:
        root = a.worker.resolve()
        verify(root)
        sys.path.insert(0, str(root / "implementation"))
        from mesh_swiglu_sdk import run

        run(root)
