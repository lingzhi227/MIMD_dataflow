"""Precision-matched control retaining the pinned source Prefill engine.

The existing executed source control supplies topology, callbacks and half
kernels. Only the explicitly selected mixed stages change precision, using the
same shared f32 math/communication libraries as the HLS lowering. This compares
source versus compiler composition overhead, not independent library arithmetic.
"""

import datetime, difflib, hashlib, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from input_attention_mixed_csl import function
from probe_runtime import verify, sha


def prepare():
    source = ROOT / "evidence/input-attention-source-20260907T122338297913Z"
    typed = ROOT / "evidence/input-attention-codegen-20260907T142401616614Z"
    verify(source)
    verify(typed)
    root = (
        ROOT
        / "evidence"
        / (
            "input-attention-mixed-source-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    print(root.relative_to(ROOT), flush=True)
    provenance = json.loads((source / "provenance.json").read_text())
    for name in provenance["files"]:
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, dest)
    text = (root / "prefill.csl").read_text()
    before = text
    assert (root / "comm_lib/comm_pe.csl").read_bytes() == (
        typed / "inference_comm.csl"
    ).read_bytes()
    shutil.copyfile(
        typed / "inference_comm_wide_rows.csl", root / "comm_lib/comm_pe.csl"
    )
    for name in ("softmax_f32_local.csl", "rms_f32_local.csl"):
        shutil.copyfile(typed / name, root / name)

    def replace_fn(name, new):
        nonlocal text
        a, b, _ = function(text, name)
        text = text[:a] + new + text[b:]

    def edit_fn(name, pairs):
        nonlocal text
        a, b, body = function(text, name)
        for old, new in pairs:
            assert body.count(old) == 1, (name, old)
            body = body.replace(old, new)
        text = text[:a] + body + text[b:]

    edit_fn(
        "matmul_compute",
        [
            (
                "fn matmul_compute() void {",
                "fn matmul_compute() void {if(hls_phase==-9 or hls_phase==-4 or hls_phase==-3){source_wide_compute();return;}",
            )
        ],
    )
    edit_fn(
        "left_matrix_shift_callback",
        [
            (
                "comm_mod.left_matrix_shift(ptr_left_matrix_send, ptr_left_matrix_recv, left_dim_length, shift_round);",
                "if(hls_phase==-4 or hls_phase==-3){comm_mod.left_matrix_shift_f32(ptr_left_matrix_send,ptr_left_matrix_recv,left_dim_length,shift_round);}else{comm_mod.left_matrix_shift(ptr_left_matrix_send, ptr_left_matrix_recv, left_dim_length, shift_round);}",
            )
        ],
    )
    edit_fn(
        "value_shift",
        [
            (
                "comm_mod.mm_two_hop_comm_T(ptr_right_matrix_send,ptr_right_matrix_recv,value_step);",
                "comm_mod.mm_two_hop_comm_T_f32(ptr_right_matrix_send,ptr_right_matrix_recv,value_step);",
            )
        ],
    )
    edit_fn(
        "xv_matmul",
        [
            ("@fmovh(comp_dest_dsr_1, 0.0);", "@fmovs(source_v_dsd,0.0);"),
            (
                "ptr_out_matrix = &XV_tile;",
                "ptr_out_matrix = @ptrcast([*]f16,&source_v);",
            ),
        ],
    )
    edit_fn(
        "output_matmul",
        [
            (
                "@fmovh(comp_dest_dsr_1, 0.0);",
                "@fmovs(source_a_dsd,0.0);@fmovs(source_work0_dsd,source_v_dsd);",
            ),
            (
                "ptr_left_matrix_send = &seqLen_seqLen_tmp;",
                "ptr_left_matrix_send = @ptrcast([*]f16,&source_exponents);",
            ),
            (
                "ptr_left_matrix_recv = &score;",
                "ptr_left_matrix_recv = @ptrcast([*]f16,&source_probability);",
            ),
            (
                "ptr_right_matrix_send = &seqLen_dim_tmp;",
                "ptr_right_matrix_send = @ptrcast([*]f16,&source_work1);",
            ),
            (
                "ptr_right_matrix_recv = &XV_tile;",
                "ptr_right_matrix_recv = @ptrcast([*]f16,&source_work0);",
            ),
            (
                "ptr_out_matrix = &output_tile;",
                "ptr_out_matrix = @ptrcast([*]f16,&source_a);",
            ),
        ],
    )
    edit_fn(
        "h1_matmul",
        [
            (
                "@fmovh(comp_dest_dsr_1, 0.0);",
                "@fmovs(source_projection_dsd,0.0);@fmovs(source_work0_dsd,source_a_dsd);",
            ),
            (
                "ptr_left_matrix_send = &seqLen_dim_tmp;",
                "ptr_left_matrix_send = @ptrcast([*]f16,&source_work1);",
            ),
            (
                "ptr_left_matrix_recv = &output_tile;",
                "ptr_left_matrix_recv = @ptrcast([*]f16,&source_work0);",
            ),
            (
                "ptr_out_matrix = &h1_tile;",
                "ptr_out_matrix = @ptrcast([*]f16,&source_projection);",
            ),
        ],
    )
    replace_fn(
        "softmax_score",
        """fn softmax_score() void {
 source_convert.convert(f32,f16,source_probability_dsd,score_dsd,.{});
 source_softmax.scale_max(&source_probability,&source_peaks,@as(f32,alpha));
 comm_mod.mv_allreduce_max_x_f32(&source_peaks);
 source_softmax.shift_exp(&source_probability,&source_peaks,&source_exponents);
 source_softmax.local_sum(&source_exponents,&source_rows);comm_mod.mv_allreduce_add_x_f32(&source_rows);
 source_softmax.normalize(&source_probability,&source_exponents,&source_rows);
 @fmovs(source_probability_snapshot_dsd,source_probability_dsd);
 source_convert.convert(f16,f32,score_dsd,source_probability_dsd,.{});prefill_struct();
}""",
    )
    replace_fn(
        "z_add",
        """fn z_add() void {
 source_convert.convert(f32,f16,source_work0_dsd,X_dsd,.{});@fadds(source_z_dsd,source_projection_dsd,source_work0_dsd);
 source_convert.convert(f16,f32,Z_dsd,source_z_dsd,.{});prefill_struct();
}""",
    )
    replace_fn(
        "rmsnorm_z",
        """fn rmsnorm_z() void {
 source_convert.convert(f32,f16,source_gamma_dsd,W_dsd,.{});
 source_rms.square_sum(&source_z,&source_work1,&source_rows);comm_mod.mv_allreduce_add_x_f32(&source_rows);
 source_rms.inverse(&source_rows);source_rms.normalize(&source_z,&source_gamma,&source_normalized,&source_rows);
 source_convert.convert(f16,f32,@set_dsd_base_addr(Z_dsd,ptr_Z_norm),source_normalized_dsd,.{});prefill_struct();
}""",
    )
    replace_fn(
        "add_result",
        """fn add_result() void {
 source_convert.convert(f32,f16,source_work0_dsd,h2_dsd,.{});@fadds(source_work0_dsd,source_z_dsd,source_work0_dsd);
 source_convert.convert(f16,f32,Z_dsd,source_work0_dsd,.{});
}""",
    )
    for phase, half, wide in [
        (-9, "XV_dsd", "source_v_dsd"),
        (-4, "output_dsd", "source_a_dsd"),
        (-3, "h1_dsd", "source_projection_dsd"),
    ]:
        marker = f"else if(hls_phase=={phase}){{"
        assert text.count(marker) == 1
        text = text.replace(
            marker, marker + f"source_convert.convert(f16,f32,{half},{wide},.{{}});"
        )
    text += """
const source_convert=@import_module("<dsd_ops>");
const source_softmax=@import_module("softmax_f32_local.csl",.{.rows=seq_len_p_pe,.cols=seq_len_p_pe});
const source_rms=@import_module("rms_f32_local.csl",.{.rows=seq_len_p_pe,.features=dim_p_pe,.global_features=dim_p_pe*P,.epsilon=@as(f32,0.000001)});
fn source_map_half(v:f16) void {@fmacs(comp_dest_dsr_1,comp_src0_dsr_1,comp_src1_dsr_1,@as(f32,v));}
fn source_map_float(v:f32) void {@fmacs(comp_dest_dsr_1,comp_src0_dsr_1,comp_src1_dsr_1,v);}
fn source_wide_compute() void {
 swap_ptr=ptr_left_matrix_send;ptr_left_matrix_send=ptr_left_matrix_recv;ptr_left_matrix_recv=swap_ptr;
 swap_ptr=ptr_right_matrix_send;ptr_right_matrix_send=ptr_right_matrix_recv;ptr_right_matrix_recv=swap_ptr;
 if(step<P){
  comm_mod.mixed_mm(ptr_left_matrix_send,ptr_right_matrix_send,ptr_left_matrix_recv,ptr_right_matrix_recv,step,Mt,Kt,Nt,hls_phase!=-9,hls_phase==-4);
  var lh=@get_dsd(mem1d_dsd,.{.base_address=ptr_left_matrix_send,.extent=@as(u16,Mt)});
  var lf=@get_dsd(mem1d_dsd,.{.base_address=@ptrcast([*]f32,ptr_left_matrix_send),.extent=@as(u16,Mt)});
  var rh=@get_dsd(mem1d_dsd,.{.base_address=ptr_right_matrix_send,.extent=@as(u16,Nt)});
  var rf=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{dim_p_pe}->source_work0[i*seq_len_p_pe]});rf=@set_dsd_base_addr(rf,@ptrcast([*]f32,ptr_right_matrix_send));
  const out=@get_dsd(mem1d_dsd,.{.base_address=@ptrcast([*]f32,ptr_out_matrix),.extent=@as(u16,Mt)});
  for(@range(i16,Kt)) |k| {
   if(hls_phase==-9){source_convert.convert(f32,f16,source_column_dsd,lh,.{});}
   @load_to_dsr(comp_dest_dsr_1,out,.{.save_address=true});@load_to_dsr(comp_src0_dsr_1,out,.{.save_address=true});
   if(hls_phase==-9){@load_to_dsr(comp_src1_dsr_1,source_column_dsd,.{.save_address=false});}else{@load_to_dsr(comp_src1_dsr_1,lf,.{.save_address=false});}
   if(hls_phase==-4){@map(source_map_float,rf);rf=@increment_dsd_offset(rf,1,f32);}else{@map(source_map_half,rh);rh=@increment_dsd_offset(rh,Nt,f16);}
   lh=@increment_dsd_offset(lh,Mt,f16);lf=@increment_dsd_offset(lf,Mt,f32);
  }
  step+=1;@activate(next_step_id);
 }else{step=0;prefill_struct();}
}
"""
    lengths = {
        name: 64
        for name in (
            "v",
            "a",
            "projection",
            "z",
            "normalized",
            "probability",
            "exponents",
            "probability_snapshot",
            "work0",
            "work1",
        )
    }
    lengths.update(gamma=8, rows=8, peaks=8, column=8)
    exports = {
        name: 64
        for name in ("v", "a", "projection", "z", "normalized", "probability_snapshot")
    }
    for name, length in lengths.items():
        text += f"var source_{name}=@zeros([{length}]f32);const source_{name}_dsd=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{{length}}}->source_{name}[i]}});\n"
        if name in exports:
            text += f'var source_{name}_ptr:[*]f32=&source_{name};comptime {{@export_symbol(source_{name}_ptr,"source_{name}");}}\n'
    (root / "prefill.csl").write_text(text)
    (root / "mixed-source-adapter.diff").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                text.splitlines(True),
                fromfile="executed-source-half/prefill.csl",
                tofile="precision-matched-source/prefill.csl",
            )
        )
    )
    layout = (root / "layout.csl").read_text()
    i = layout.rfind("}")
    layout = (
        layout[:i]
        + "".join(f'@export_name("source_{name}",[*]f32,true);' for name in exports)
        + layout[i:]
    )
    (root / "layout.csl").write_text(layout)
    schema = json.loads((root / "schema.json").read_text())
    schema["outputs"].update({"source_" + k: v for k, v in exports.items()})
    schema.setdefault("output_word_bits", {}).update(
        {"source_" + k: 32 for k in exports}
    )
    (root / "schema.json").write_text(json.dumps(schema) + "\n")
    physical = json.loads((typed / "inputs.json").read_text())
    for b in physical:
        b["input_x"] = b.pop("residual")
    assert physical[:3] == json.loads((source / "inputs.json").read_text())
    (root / "inputs.json").write_text(json.dumps(physical) + "\n")
    shutil.copyfile(typed / "logical-inputs.json", root / "logical-inputs.json")
    shutil.copyfile(__file__, root / "mixed-source-control-driver.py")
    provenance.update(
        precision_matched_mixed=True,
        scope=__doc__,
        parent_source_provenance_sha256=sha(source / "provenance.json"),
        typed_csl_reference=str(typed),
        source_unchanged_half_engine=True,
    )
    provenance["files"] = {
        str(p.relative_to(root)): sha(p)
        for p in root.rglob("*")
        if p.is_file() and p.name != "provenance.json"
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return root


if __name__ == "__main__":
    prepare()
