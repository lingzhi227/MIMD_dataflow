"""Precision/order-matched pinned Decode normalized projection control; numerical source remains visible."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, difflib, json, shutil, sys
from pathlib import Path
from probe_runtime import execute, mesh_half_worker, sha, read

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]


def prepare(bundle):
    sys.path.insert(0,str(bundle/"implementation"))
    from integrity import verify_bundle
    from mesh_batched_fanout_sdk import packed

    verify_bundle(bundle)
    s, m, bs = (
        read(bundle / n) for n in ("schedule.json", "semantic.json", "batches.json")
    )
    assert s["profile"] == "mesh_batched_fanout.v1"
    assert s["epsilon"] == 1e-6 and s["instrumentation"] == "sampled"
    assert s["projections"] == 2 or s["F"] == s["N"]
    c = s["projections"]
    weight_names = ["Q", "K", "V"] if c == 3 else ["UP", "GATE"]
    slab = "QKV" if c == 3 else "ZZ"
    calls = (
        "xq_matvec_mult();xk_matvec_mult();xv_matvec_mult();"
        if c == 3
        else "up_matvec_mult();gate_matvec_mult();"
    )
    fusion = (
        "all_reduce_bsz_dim_QKV_fusion"
        if c == 3
        else "all_reduce_bsz_ffn_dim_ZZ_fusion"
    )
    root = (
        ROOT
        / "validation/evidence"
        / (
            "batched-fanout-source-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    src = ROOT / "third_party/sources/waferllm/Decode/src"
    shutil.copytree(src / "comm_lib", root / "comm_lib")
    original = (src / "decode.csl").read_text()
    text = original[: original.index("\ncomptime {")]
    # Preserve logical odd batch for fused projection extents. Only the RMS
    # collective gets one zero padding half lane; its original DSDs are separate.
    comm = root / "comm_lib/comm_pe.csl"
    ct = comm.read_text()
    import re

    ct, count = re.subn(
        r"(const \w+_bsz = @get_dsd\((?:fabin|fabout)_dsd, \.\{\s*\.extent = )bsz,",
        r"\g<1>((bsz+1)/2)*2,",
        ct,
    )
    assert count == 10, count
    ct = ct.replace(
        "var vector_buf_dsd_bsz = @get_dsd(mem1d_dsd, .{ .base_address = &dummy, .extent = bsz });",
        "var vector_buf_dsd_bsz = @get_dsd(mem1d_dsd, .{.base_address=&dummy,.extent=((bsz+1)/2)*2});",
    )
    assert ".extent=((bsz+1)/2)*2" in ct
    comm.write_text(ct)
    if c == 2:
        for fn, nextfn in [
            ("up_matvec_mult", "gate_matvec_mult"),
            ("gate_matvec_mult", "z2_silu"),
        ]:
            a = text.index("fn " + fn + "()")
            b = text.index("fn " + nextfn + "()", a)
            text = text[:a] + text[a:b].replace("ptr_z_norm", "ptr_X_norm") + text[b:]
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
    # Keep original matvec functions and fused collective; expose all branch
    # partials and the original packed result for an exact schedule comparison.
    text = text.replace(
        " rmsnorm_x();\n timestamp.get_timestamp",
        " rmsnorm_x();\n "
        + calls
        + "\n const sd=@get_dsd(mem1d_dsd,.{.base_address=ptr_"
        + slab
        + "_tile,.extent="
        + str(s["packed_length"])
        + "});\n const hd=@get_dsd(mem1d_dsd,.{.base_address=&partial,.extent="
        + str(s["packed_length"])
        + "});@mov16(hd,sd);\n comm_mod."
        + fusion
        + "(py,quotient_y,remainder_y,ptr_"
        + slab
        + "_tile);\n timestamp.get_timestamp",
        1,
    )
    extra = (
        "var partial=@zeros(["
        + str(s["packed_length"])
        + ']f16);var partial_ptr:[*]f16=&partial;\ncomptime {@export_symbol(partial_ptr,"partial");@export_symbol(ptr_'
        + slab
        + '_tile,"projections");'
    )
    extra += (
        "".join(
            "@export_symbol(ptr_" + name + '_weight,"weight' + str(i) + '");'
            for i, name in enumerate(weight_names)
        )
        + "}\n"
    )
    text += extra
    (root / "decode.csl").write_text(text)
    layout = """param P:i16;param bsz:i16;param dim_p_pe:i16;param groups:i16;param Ft:i16;
const memcpy=@import_module("<memcpy/get_params>",.{.width=P,.height=P});
layout {@set_rectangle(P,P);for(@range(i16,P)) |y| {for(@range(i16,P)) |x| {
 @set_tile_code(x,y,"decode.csl",.{.memcpy_params=memcpy.get_params(x),.P=P,.bsz=bsz,.dim_p_pe=dim_p_pe,.pes_p_head=P,.pes_p_kv_head=P,.head_dim_p_pe=dim_p_pe,.seq_len_p_pe=8,.ffn_dim_p_pe=Ft,.pe_num_p_group=P/groups,.root_1st_phase=(P/groups)/2,.root_2nd_phase=(groups/2)*(P/groups)+(P/groups)/2,.reduce_1st_color_0=@get_color(8),.reduce_1st_color_1=@get_color(7),.reduce_2nd_color_0=@get_color(6),.reduce_2nd_color_1=@get_color(5),.broadcast_color=@get_color(9)});
 }}
 @export_name("X",[*]f16,true);@export_name("W",[*]f16,true);@export_name("result",[*]f16,true);@export_name("sums",[*]f16,true);@export_name("history",[*]f16,true);@export_name("progress",[*]u16,true);@export_name("timing",[*]u16,true);@export_name("init_task",fn()void);@export_name("hls_main",fn()void);
}
"""
    exports = (
        ' @export_name("partial",[*]f16,true);@export_name("projections",[*]f16,true);'
        + "".join('@export_name("weight' + str(i) + '",[*]f16,true);' for i in range(c))
    )
    layout = (
        layout[: layout.rfind("}")] + exports + "\n}" + layout[layout.rfind("}") + 1 :]
    )
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
    ports.update(
        partial=s["packed_length"],
        projections=s["packed_length"],
        **{"weight" + str(i): s["Nt"] * s["Ft"] for i in range(c)},
    )

    def pack_control(b):
        data = packed(s, m, b)
        weights = data.pop("weights")
        extent = s["Nt"] * s["Ft"]
        data.update(
            {
                "weight" + str(i): weights[:, :, i * extent : (i + 1) * extent]
                for i in range(c)
            }
        )
        return {k: v.astype(float).tolist() for k, v in data.items()}

    values = {
        "logical-inputs.json": bs,
        "inputs.json": [pack_control(b) for b in bs],
        "schema.json": dict(
            rows=s["P"],
            cols=s["P"],
            inputs=dict(
                X=s["length"],
                W=s["Nt"],
                **{"weight" + str(i): s["Nt"] * s["Ft"] for i in range(c)},
            ),
            outputs=ports,
            immutable=["X", "W"] + ["weight" + str(i) for i in range(c)],
            progress="progress",
            initialize="init_task",
            launch="hls_main",
        ),
        "runtime-options.json": dict(
            suppress_trace=True, num_threads=8, dump_core=False
        ),
        "sdk-command.json": [
            "cslc",
            "layout.csl",
            "--arch=wse3",
            f"--fabric-dims={s['P']+7},{s['P']+2}",
            "--fabric-offsets=4,1",
            f"--params=P:{s['P']},bsz:{s['B']},dim_p_pe:{s['Nt']},groups:{s['groups']},Ft:{s['Ft']}",
            "-o=out",
            "--memcpy",
            "--channels=1",
        ],
    }
    for name, v in values.items():
        (root / name).write_text(json.dumps(v, indent=2) + "\n")
    shutil.copyfile(__file__, root / "driver.py")
    shutil.copyfile(ROOT / "experiments/probe_runtime.py", root / "probe_runtime.py")
    shutil.copyfile(ROOT / "lib/Runtime/sdk_process.py", root / "sdk_process.py")
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                hls_bundle=str(bundle.relative_to(ROOT)),
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                source_files={
                    str(p.relative_to(src)): sha(p) for p in src.rglob("*.csl")
                },
                adaptation="Original Decode RMS and two/three vecmat functions plus original fused collective. Correct RMS recurrence/input, pad only RMS DSD extents, UP/GATE consumes the normalized input subgraph directly. All branch partials observed. No HLS local modules substituted; no full Decode claim.",
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
        execute(a.execute.resolve(), 7200)
