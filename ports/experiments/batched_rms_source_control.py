"""Precision/order-matched pinned Decode RMS control; numerical source remains visible."""

import argparse, datetime, difflib, json, shutil, sys
from pathlib import Path
from probe_runtime import execute, mesh_half_worker, sha, read

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]


def prepare(bundle):
    from integrity import verify_bundle
    from mesh_batched_rms_sdk import packed

    verify_bundle(bundle)
    s, m, bs = (
        read(bundle / n) for n in ("schedule.json", "semantic.json", "batches.json")
    )
    assert s["profile"] == "mesh_batched_rms.v1"
    root = (
        ROOT
        / "evidence"
        / (
            "batched-rms-source-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    src = ROOT / "projects/waferllm/upstream/Decode/src"
    shutil.copytree(src / "comm_lib", root / "comm_lib")
    original = (src / "decode.csl").read_text()
    text = original[: original.index("\ncomptime {")]
    text = text.replace(
        ".P = P, .bsz = bsz, .dim_p_pe = dim_p_pe,",
        ".P = P, .bsz = ((bsz+1)/2)*2, .dim_p_pe = dim_p_pe,",
        1,
    )
    text = text.replace(
        "var local_sum: [bsz]f16 = @zeros([bsz]f16);",
        "var local_sum: [((bsz+1)/2)*2]f16 = @zeros([((bsz+1)/2)*2]f16);",
    )
    start = text.index("fn rmsnorm_x() void {")
    end = text.index("\nfn xq_matvec_mult()", start)
    body = text[start:end]
    a = body.index(
        "    dim_p_pe_dsd_1 = @set_dsd_base_addr(dim_p_pe_dsd_1, ptr_X_tmp);"
    )
    b = body.index("    comm_mod.all_reduce_bsz", a)
    body = body[:a] + """
 const acc=@get_dsd(mem1d_dsd,.{.base_address=&control_acc,.extent=1});
 const element=@get_dsd(mem1d_dsd,.{.base_address=ptr_X_tmp,.extent=1});
 @load_to_dsr(dest_dsr_2,acc,.{.save_address=false});
 @load_to_dsr(src0_dsr_2,acc,.{.save_address=false});
 @load_to_dsr(src1_dsr_1,element,.{.save_address=true});
 for(@range(i16,bsz)) |i| {
  @fmovh(dest_dsr_2,0.0);
  for(@range(i16,dim_p_pe)) |j| {@faddh(dest_dsr_2,src0_dsr_2,src1_dsr_1);}
  local_sum[i]=control_acc[0];
 }
 @mov16(control_hist_dsd,control_sum_dsd);
""" + body[b:]
    call = "    comm_mod.all_reduce_bsz(py, quotient_y, remainder_y, ptr_local_sum);"
    assert body.count(call) == 1
    body = body.replace(
        call,
        call
        + "\n const h=@increment_dsd_offset(control_hist_dsd,control_padded,f16);@mov16(h,control_sum_dsd);",
    )
    needle = "dim_p_pe_dsd_2 = @set_dsd_base_addr(dim_p_pe_dsd_2, ptr_X_tmp);"
    assert body.count(needle) == 1
    body = body.replace(needle, needle.replace("ptr_X_tmp", "ptr_X"))
    text = text[:start] + body + text[end:] + """
const control_padded:i16=((bsz+1)/2)*2;
var control_acc=@zeros([1]f16);var history=@zeros([2*control_padded]f16);
var progress=@zeros([1]u16);var timing=@zeros([6]u16);
var control_start=@zeros([3]u16);var control_end=@zeros([3]u16);
const control_sum_dsd=@get_dsd(mem1d_dsd,.{.base_address=&local_sum,.extent=@as(u16,control_padded)});
const control_hist_dsd=@get_dsd(mem1d_dsd,.{.base_address=&history,.extent=@as(u16,control_padded)});
fn hls_main() void {
 timestamp.enable_tsc();timestamp.get_timestamp(&control_start);
 @fmovh(control_sum_dsd,0.0);
 rmsnorm_x();
 timestamp.get_timestamp(&control_end);timestamp.disable_tsc();
 for(@range(i16,3)) |i| {timing[i]=control_start[i];timing[i+3]=control_end[i];}
 progress[0]+=1;sys_mod.unblock_cmd_stream();
}
var hp:[*]f16=&history;var pp:[*]u16=&progress;var tp:[*]u16=&timing;
comptime {@export_symbol(ptr_X,"X");@export_symbol(ptr_W,"W");@export_symbol(ptr_X_norm,"result");@export_symbol(ptr_local_sum,"sums");@export_symbol(hp,"history");@export_symbol(pp,"progress");@export_symbol(tp,"timing");@export_symbol(init_task);@export_symbol(hls_main);}
"""
    (root / "decode.csl").write_text(text)
    layout = """param P:i16;param bsz:i16;param dim_p_pe:i16;param groups:i16;
const memcpy=@import_module("<memcpy/get_params>",.{.width=P,.height=P});
layout {@set_rectangle(P,P);for(@range(i16,P)) |y| {for(@range(i16,P)) |x| {
 @set_tile_code(x,y,"decode.csl",.{.memcpy_params=memcpy.get_params(x),.P=P,.bsz=bsz,.dim_p_pe=dim_p_pe,.pes_p_head=P,.pes_p_kv_head=P,.head_dim_p_pe=dim_p_pe,.seq_len_p_pe=8,.ffn_dim_p_pe=8,.pe_num_p_group=P/groups,.root_1st_phase=(P/groups)/2,.root_2nd_phase=(groups/2)*(P/groups)+(P/groups)/2,.reduce_1st_color_0=@get_color(8),.reduce_1st_color_1=@get_color(7),.reduce_2nd_color_0=@get_color(6),.reduce_2nd_color_1=@get_color(5),.broadcast_color=@get_color(9)});
 }}
 @export_name("X",[*]f16,true);@export_name("W",[*]f16,true);@export_name("result",[*]f16,true);@export_name("sums",[*]f16,true);@export_name("history",[*]f16,true);@export_name("progress",[*]u16,true);@export_name("timing",[*]u16,true);@export_name("init_task",fn()void);@export_name("hls_main",fn()void);
}
"""
    (root / "layout.csl").write_text(layout)
    (root / "source-adapter.diff").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                text.splitlines(True),
                fromfile="pinned/Decode/decode.csl",
                tofile="control/decode.csl",
            )
        )
    )
    ports = dict(
        X=s["length"],
        W=s["Nt"],
        result=s["length"],
        sums=s["padded_batches"],
        history=2 * s["padded_batches"],
        progress=1,
        timing=6,
    )
    values = {
        "logical-inputs.json": bs,
        "inputs.json": [
            {k: v.astype(float).tolist() for k, v in packed(s, m, b).items()}
            for b in bs
        ],
        "schema.json": dict(
            rows=s["P"],
            cols=s["P"],
            inputs=dict(X=s["length"], W=s["Nt"]),
            outputs=ports,
            immutable=["X", "W"],
            progress="progress",
            initialize="init_task",
            launch="hls_main",
        ),
        "runtime-options.json": dict(
            suppress_trace=True, num_threads=8, dump_core=True
        ),
        "sdk-command.json": [
            "cslc",
            "layout.csl",
            "--arch=wse3",
            f"--fabric-dims={s['P']+7},{s['P']+2}",
            "--fabric-offsets=4,1",
            f"--params=P:{s['P']},bsz:{s['B']},dim_p_pe:{s['Nt']},groups:{s['groups']}",
            "-o=out",
            "--memcpy",
            "--channels=1",
        ],
    }
    for name, v in values.items():
        (root / name).write_text(json.dumps(v, indent=2) + "\n")
    shutil.copyfile(__file__, root / "driver.py")
    shutil.copyfile(ROOT / "experiments/probe_runtime.py", root / "probe_runtime.py")
    shutil.copyfile(ROOT / "toolchain/sdk_process.py", root / "sdk_process.py")
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                hls_bundle=str(bundle.relative_to(ROOT)),
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                source_files={
                    str(p.relative_to(src)): sha(p) for p in src.rglob("*.csl")
                },
                adaptation="RMS-only exports; original Decode local norm and original grouped communication. Repair sum recurrence with memory DSR, normalize X, pad collective to even half lanes; capture local/reduced vectors and cycle intervals. No shared HLS local RMS implementation is substituted.",
                files={
                    str(p.relative_to(root)): sha(p)
                    for p in root.rglob("*")
                    if p.is_file()
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", type=Path)
    p.add_argument("--worker", type=Path)
    p.add_argument("--execute", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.prepare.resolve())
    elif a.worker:
        mesh_half_worker(a.worker.resolve())
    elif a.execute:
        execute(a.execute.resolve(), 1200)
