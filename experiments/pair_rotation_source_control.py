"""Isolate original vector pair transform with matched HLS transport and diagnostics."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, shutil, difflib, sys
from pathlib import Path
from probe_runtime import read, sha, verify, execute

ROOT = repository_root(__file__)


def prepare(hls, calls=3):
    s = read(hls / "schedule.json")
    assert (
        s["profile"] == "mesh_pair_rotation.v1"
        and s["broadcast_coefficients"]
        and s["pair_order"] == "odd_even"
    )
    root = (
        ROOT
        / "validation/evidence"
        / (
            "pair-rotation-source-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    src = ROOT / "third_party/sources/waferllm/Prefill/src/prefill.csl"
    text = src.read_text()
    original = text[text.index("fn xq_rope()") : text.index("fn xk_rope()")]
    body = original
    needle = "        @fsubh(seqLen_dsd_1, X_tmp_1_dsd, X_tmp_2_dsd);"
    assert body.count(needle) == 1
    sampling = """        if(sampled!=0){
          @fmovh(@increment_dsd_offset(hd,(4*i)*Mt,f16),X_tmp_1_dsd);
          @fmovh(@increment_dsd_offset(hd,(4*i+1)*Mt,f16),X_tmp_2_dsd);
          @fmovh(@increment_dsd_offset(hd,(4*i+2)*Mt,f16),X_tmp_3_dsd);
          @fmovh(@increment_dsd_offset(hd,(4*i+3)*Mt,f16),X_tmp_4_dsd);
        }
"""
    body = body.replace(needle, sampling + needle)
    pe = """param memcpy_params;param Mt:i16;param Nt:i16;param broadcast:i16;param swapped:i16;param sampled:i16;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);const time=@import_module("<time>");
const seq_len_p_pe:i16=Mt;const seq_len_p_pe_2:i16=2*Mt;const _dim_p_pe:i16=Nt;
var x=@zeros([Mt*Nt]f16);var ptr_XQ:[*]f16=&x;
var freqs_cos=@zeros([Nt/2]f16);var freqs_sin=@zeros([Nt/2]f16);var cos_val:f16=0.0;var sin_val:f16=0.0;
var seqLen_dsd_1=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{Mt}->x[i]});
var seqLen_dsd_2=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{Mt}->x[i]});
var history=@zeros([if(sampled!=0) 2*Mt*Nt else 1]f16);const hd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{Mt}->history[i]});
"""
    for k in range(1, 5):
        pe += f"var X_tmp_{k}=@zeros([Mt]f16);var X_tmp_{k}_dsd=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{Mt}}->X_tmp_{k}[i]}});\n"
    pe += (
        body
        + """var progress=@zeros([3]u16);var timing=@zeros([6]u16);var start=@zeros([3]u16);var end=@zeros([3]u16);
fn prefill_struct() void {}
fn hls_main() void {
 time.enable_tsc();time.get_timestamp(&start);progress[0]=0;progress[1]=0;
 xq_rope();progress[0]=Nt/2;progress[1]=1;progress[2]+=1;
 time.get_timestamp(&end);time.disable_tsc();for(@range(i16,3)) |i| {timing[i]=start[i];timing[i+3]=end[i];}sys.unblock_cmd_stream();
}
var cp:[*]f16=&freqs_cos;var sp:[*]f16=&freqs_sin;var hp:[*]f16=&history;var pp:[*]u16=&progress;var tp:[*]u16=&timing;
comptime {@export_symbol(ptr_XQ,"x");@export_symbol(cp,"cosine");@export_symbol(sp,"sine");@export_symbol(ptr_XQ,"result");@export_symbol(hp,"history");@export_symbol(pp,"progress");@export_symbol(tp,"timing");@export_symbol(hls_main);}
"""
    )
    (root / "pe.csl").write_text(pe)
    (root / "original-kernel.csl").write_text(original)
    (root / "source-body-observation.diff").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                body.splitlines(True),
                fromfile="original",
                tofile="observed",
            )
        )
    )
    for n in [
        "layout.csl",
        "schedule.json",
        "semantic.json",
        "runtime-options.json",
        "WaferLLM-LICENSE.txt",
    ]:
        shutil.copyfile(hls / n, root / n)
    (root / "batches.json").write_text(
        json.dumps(read(hls / "batches.json")[:calls]) + "\n"
    )
    shutil.copytree(hls / "implementation", root / "implementation")
    shutil.copyfile(ROOT / "experiments/probe_runtime.py", root / "probe_runtime.py")
    shutil.copyfile(ROOT / "lib/Runtime/sdk_process.py", root / "sdk_process.py")
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
                source_sha256=sha(src),
                files=files,
                scope="Original xq_rope arithmetic/DSD loop with optional product observations and empty outer continuation. Temporary DSDs explicitly use Mt, repairing observed source length mismatch when Mt!=Nt/2. Source input overwritten in place; HLS inputs immutable. Matched transport and geometry; no full-model claim.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--calls", type=int, choices=(3, 6), default=3)
    p.add_argument("--prepare", type=Path)
    p.add_argument("--execute", type=Path)
    p.add_argument("--worker", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.prepare.resolve(), a.calls)
    elif a.execute:
        execute(a.execute.resolve(), 1800)
    elif a.worker:
        root = a.worker.resolve()
        verify(root)
        sys.path.insert(0, str(root / "implementation"))
        from mesh_pair_rotation_sdk import run

        run(root)
