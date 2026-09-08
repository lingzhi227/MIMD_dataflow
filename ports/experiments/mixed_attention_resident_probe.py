"""Experimental CSL mixed-width resident path, pending typed HLS admission.

Build on the executed generated31-node path, preserve all inputs and failures.
Probe f32 probability, V/PV/O/Z and RMS(Z), half Q/K and blocked-half MLP.
"""

import argparse, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain"), str(ROOT / "experiments")]
from input_attention_codegen_probe import prepare
from softmax_f32_resident_probe import adapt as softmax_adapt


def function(text, name):
    a = text.index("fn " + name + "(")
    start = text.index("{", a)
    depth = 1
    i = start + 1
    while depth:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return a, i, text[a:i]


def native_gate(root):
    from input_attention_mixed_precision_study import HELPERS
    from native_transport import parse_outputs
    from input_attention_fixtures import check

    helper = HELPERS + r"""
namespace precision_experiment {
template<int M,int N>auto softmax(const spatial::tensor<M,N,spatial::f16>& x,float scale){
 spatial::tensor<M,N,float> out;
 for(int i=0;i<M;++i){
  float peak=-std::numeric_limits<float>::infinity();
  for(int j=0;j<N;++j)peak=std::max(peak,float(x.data[i*N+j])*scale);
  float sum=0;
  for(int j=0;j<N;++j){out.data[i*N+j]=std::exp(float(x.data[i*N+j])*scale-peak);sum+=out.data[i*N+j];}
  for(int j=0;j<N;++j)out.data[i*N+j]=out.data[i*N+j]/sum;
 }return out;
}
}
"""
    text = (
        (root / "observed.cpp")
        .read_text()
        .replace('#include "spatial.hpp"', '#include "spatial.hpp"\n' + helper)
    )
    for a, b in (
        ("input_normalized", "v_weight"),
        ("probability", "v"),
        ("attention", "output_weight"),
    ):
        old = f"spatial::matmul({a},{b})"
        assert text.count(old) == 1
        text = text.replace(old, f"precision_experiment::product<float>({a},{b})")
    for old, new in (
        (
            "spatial::softmax(score,0.125)",
            "precision_experiment::softmax(score,0.125f)",
        ),
        (
            "spatial::add(projection,input_x)",
            "precision_experiment::add<float>(projection,input_x)",
        ),
        (
            "spatial::rmsnorm(z,gamma,0.000001)",
            "precision_experiment::norm(z,gamma,0.000001)",
        ),
        ("spatial::add(z,delta)", "precision_experiment::add<spatial::f16>(z,delta)"),
    ):
        assert text.count(old) == 1
        text = text.replace(old, new)
    (root / "mixed-native.cpp").write_text(text)
    cmd = json.loads((root / "observed-command.json").read_text())
    cmd[cmd.index(str(root / "observed.cpp"))] = str(root / "mixed-native.cpp")
    cmd[-1] = str(root / "mixed-native")
    (root / "mixed-native-command.json").write_text(json.dumps(cmd) + "\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    (root / "mixed-native-compile.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0
    result = subprocess.run(
        [cmd[-1]],
        input=(root / "native-input.txt").read_text(),
        capture_output=True,
        text=True,
    )
    (root / "mixed-native-output.txt").write_text(result.stdout)
    (root / "mixed-native-stderr.txt").write_text(result.stderr)
    assert result.returncode == 0
    batches = json.loads((root / "logical-inputs.json").read_text())
    rows = parse_outputs(result.stdout)
    assert len(rows) == len(batches) == 8
    checks = []
    for b, row in zip(batches, rows):
        observations = {k[10:]: v for k, v in row.items() if k.startswith("__observe_")}
        observations["v"] = observations["v_raw"]
        checks.append(
            check(64, 64, 256, 1e-6, 0.125, b, {"output": row["output"]}, observations)
        )
    (root / "mixed-native-gate.json").write_text(
        json.dumps(
            dict(
                passed=True,
                checks=checks,
                scope="Actual experimental C++ f32 probability/V/PV/O/Z arithmetic, original11-input gates; not admitted typed HLS.",
            ),
            indent=2,
        )
        + "\n"
    )
    shutil.copyfile(
        ROOT / "experiments/input_attention_mixed_precision_study.py",
        root / "mixed-native-helper-origin.py",
    )


def adapt(root, s):
    native_gate(root)
    softmax_adapt(root, s)
    assert s["instrumentation"] == "counters" and s["M"] == s["N"] == 64
    cp = root / "inference_comm_wide_rows.csl"
    comm = cp.read_text()
    extra = ""
    _, _, body = function(comm, "mm_two_hop_comm")
    body = body[: body.index("    mm_two_hop_comm_impl(")]
    body = body.replace("fn mm_two_hop_comm(", "fn mixed_mm(").replace(
        "Nt: i16) void", "Nt: i16, left_wide:bool, right_wide:bool) void"
    )
    body += """
 const lh=@get_dsd(mem1d_dsd,.{.base_address=left_matrix_send_buffer_ptr,.extent=@as(u16,Mt*Kt)});
 const lrh=@get_dsd(mem1d_dsd,.{.base_address=left_matrix_recv_buffer_ptr,.extent=@as(u16,Mt*Kt)});
 const rh=@get_dsd(mem1d_dsd,.{.base_address=right_matrix_send_buffer_ptr,.extent=@as(u16,Kt*Nt)});
 const rrh=@get_dsd(mem1d_dsd,.{.base_address=right_matrix_recv_buffer_ptr,.extent=@as(u16,Kt*Nt)});
 const lf=@get_dsd(mem1d_dsd,.{.base_address=@ptrcast([*]f32,left_matrix_send_buffer_ptr),.extent=@as(u16,Mt*Kt)});
 const lrf=@get_dsd(mem1d_dsd,.{.base_address=@ptrcast([*]f32,left_matrix_recv_buffer_ptr),.extent=@as(u16,Mt*Kt)});
 const rf=@get_dsd(mem1d_dsd,.{.base_address=@ptrcast([*]f32,right_matrix_send_buffer_ptr),.extent=@as(u16,Kt*Nt)});
 const rrf=@get_dsd(mem1d_dsd,.{.base_address=@ptrcast([*]f32,right_matrix_recv_buffer_ptr),.extent=@as(u16,Kt*Nt)});
 if(left_wide){
  @load_to_dsr(left_send_dsr,lf);@load_to_dsr(left_recv_dsr,lrf);
  @mov32(left_matrix_out_dsr,left_send_dsr,.{.async=true,.unblock=left_matrix_finish_id,.ut_id=ut1});
  @mov32(left_recv_dsr,left_matrix_in_dsr,.{.async=true,.activate=left_matrix_finish_id,.ut_id=ut0});
 }else{
  @load_to_dsr(left_send_dsr,lh);@load_to_dsr(left_recv_dsr,lrh);
  @mov16(left_matrix_out_dsr,left_send_dsr,.{.async=true,.unblock=left_matrix_finish_id,.ut_id=ut1});
  @mov16(left_recv_dsr,left_matrix_in_dsr,.{.async=true,.activate=left_matrix_finish_id,.ut_id=ut0});
 }
 if(right_wide){
  @load_to_dsr(right_send_dsr,rf);@load_to_dsr(right_recv_dsr,rrf);
  @mov32(right_matrix_out_dsr,right_send_dsr,.{.async=true,.unblock=right_matrix_finish_id,.ut_id=ut3});
  @mov32(right_recv_dsr,right_matrix_in_dsr,.{.async=true,.activate=right_matrix_finish_id,.ut_id=ut2});
 }else{
  @load_to_dsr(right_send_dsr,rh);@load_to_dsr(right_recv_dsr,rrh);
  @mov16(right_matrix_out_dsr,right_send_dsr,.{.async=true,.unblock=right_matrix_finish_id,.ut_id=ut3});
  @mov16(right_recv_dsr,right_matrix_in_dsr,.{.async=true,.activate=right_matrix_finish_id,.ut_id=ut2});
 }
}
"""
    extra += body
    for name, impl, side, length in (
        ("left_matrix_shift", "left_matrix_shift_impl", "left", "dim_length"),
        (
            "mm_two_hop_comm_T",
            "mm_two_hop_comm_T_impl",
            "right",
            "seq_len_p_pe*dim_p_pe",
        ),
    ):
        _, _, body = function(comm, name)
        body = body[: body.index("    " + impl + "(")].replace(
            "fn " + name + "(", "fn " + name + "_f32("
        )
        body += f"""
 const send=@get_dsd(mem1d_dsd,.{{.base_address=@ptrcast([*]f32,{side}_matrix_send_buffer_ptr),.extent=@as(u16,{length})}});
 const recv=@get_dsd(mem1d_dsd,.{{.base_address=@ptrcast([*]f32,{side}_matrix_recv_buffer_ptr),.extent=@as(u16,{length})}});
 @load_to_dsr({side}_send_dsr,send);@load_to_dsr({side}_recv_dsr,recv);
 @mov32({side}_matrix_out_dsr,{side}_send_dsr,.{{.async=true,.unblock={side}_matrix_finish_id,.ut_id={'ut1' if side=='left' else 'ut3'}}});
 @mov32({side}_recv_dsr,{side}_matrix_in_dsr,.{{.async=true,.activate={side}_matrix_finish_id,.ut_id={'ut0' if side=='left' else 'ut2'}}});
}}
"""
        extra += body
    cp.write_text(comm + "\n" + extra)
    pe = (root / "pe.csl").read_text()
    pe = pe.replace(
        "fn matmul_compute() void {",
        "fn matmul_compute() void {if(phase==8 or phase==4 or phase==3){mixed_compute();return;}",
    )
    pe = pe.replace(
        "fn input_projection_setup() void {",
        "fn input_projection_setup() void {if(phase==8){mixed_v_setup();return;}",
    )
    pe = pe.replace(
        "fn setup_projection() void {",
        "fn setup_projection() void {if(phase==3){mixed_o_setup();return;}",
    )
    pe = pe.replace(
        "fn phase_finish() void {",
        "fn phase_finish() void {if(phase==4 or phase==3){mixed_finish();return;}if(phase==8){wide_ops.convert(f16,f32,attention_value_view,mixed_v_view,.{});}",
    )
    old = "comm_mod.left_matrix_shift(ptr_left_matrix_send,ptr_left_matrix_recv,left_dim_length,shift_round);"
    assert pe.count(old) == 1
    pe = pe.replace(
        old,
        "if(phase==4 or phase==3){comm_mod.left_matrix_shift_f32(ptr_left_matrix_send,ptr_left_matrix_recv,left_dim_length,shift_round);}else{"
        + old
        + "}",
    )
    a, b, _ = function(pe, "attention_value_setup")
    pe = pe[:a] + """fn attention_value_setup() void {
 @fmovs(mixed_work_view,mixed_v_view);@fmovs(mixed_a_view,0.0);
 ptr_left_matrix_send=@ptrcast([*]f16,&wide_exponents);ptr_left_matrix_recv=@ptrcast([*]f16,&wide_probability);
 ptr_right_matrix_send=@ptrcast([*]f16,&mixed_work1);ptr_right_matrix_recv=@ptrcast([*]f16,&mixed_work0);ptr_out_matrix=@ptrcast([*]f16,&mixed_a);
 Mt=seq_len_p_pe;Kt=seq_len_p_pe;Nt=dim_p_pe;left_dim_length=S;
 const px=@as(i16,layout_mod.get_x_coord());attention_remaining=if(px==0) 0 else if(px%2==0) P-px/2 else (px+1)/2;
 attention_align_step=0;attention_align=true;attention_value_align();
}
""" + pe[b:]
    old = "comm_mod.mm_two_hop_comm_T(ptr_right_matrix_send,ptr_right_matrix_recv,attention_align_step);"
    assert pe.count(old) == 1
    pe = pe.replace(old, old.replace("mm_two_hop_comm_T", "mm_two_hop_comm_T_f32"))
    marker = "wide_softmax.normalize(&wide_probability,&wide_exponents,&wide_sums);"
    assert pe.count(marker) == 1
    pe = pe.replace(
        marker, marker + "@fmovs(mixed_probability_snapshot_view,wide_values_view);"
    )
    a = pe.index(
        " @load_to_dsr(comp_dest_dsr_1,@set_dsd_base_addr(lv,@ptrcast([*]f16,&xwork)));",
        pe.index("block_accum.snapshot();"),
    )
    b = pe.index(" rms_progress[1]+=1;", a)
    pe = (
        pe[:a]
        + """ wide_ops.convert(f32,f16,mixed_work_view,@set_dsd_base_addr(lv,@ptrcast([*]f16,&xwork)),.{});
 @fadds(mixed_work_view,mixed_work_view,mixed_z_view);
 wide_ops.convert(f16,f32,@set_dsd_base_addr(lv,@ptrcast([*]f16,&xwork)),mixed_work_view,.{});
"""
        + pe[b:]
    )
    pe += DECLARATIONS
    lengths = {
        name: s["length"]
        for name in (
            "mixed_v",
            "mixed_a",
            "mixed_projection",
            "mixed_z",
            "mixed_normalized",
        )
    }
    lengths["mixed_probability_snapshot"] = s["score_length"]
    exports = ""
    for name in lengths:
        pe += f'\nvar {name}_ptr:[*]f32=&{name};comptime {{@export_symbol({name}_ptr,"{name}");}}\n'
        exports += f'@export_name("{name}",[*]f32,true);\n'
    (root / "pe.csl").write_text(pe)
    layout = (root / "layout.csl").read_text()
    a = layout.rfind("}")
    (root / "layout.csl").write_text(layout[:a] + exports + layout[a:])
    schema = json.loads((root / "schema.json").read_text())
    schema["outputs"].update(lengths)
    schema["output_word_bits"].update({k: 32 for k in lengths})
    (root / "schema.json").write_text(json.dumps(schema) + "\n")
    shutil.copyfile(
        ROOT / "toolchain/runtime/rms_f32_local.csl", root / "rms_f32_local.csl"
    )
    (root / "primitive-scope.json").write_text(
        json.dumps(
            dict(
                scope=__doc__,
                precision="f32 probabilities, V projection, PV, O projection, Z and second RMS intermediates; half normalized output into original blocked MLP. Experimental SDK path, not registered HLS.",
            ),
            indent=2,
        )
        + "\n"
    )
    shutil.copyfile(__file__, root / "mixed-adapter.py")


DECLARATIONS = """
var mixed_v=@zeros([L]f32);var mixed_a=@zeros([L]f32);var mixed_z=@zeros([L]f32);
var mixed_projection=@zeros([L]f32);var mixed_normalized=@zeros([L]f32);
var mixed_probability_snapshot=@zeros([S]f32);
var mixed_work0=@zeros([L]f32);var mixed_work1=@zeros([L]f32);var mixed_left_column=@zeros([seq_len_p_pe]f32);
var mixed_rows=@zeros([seq_len_p_pe]f32);var mixed_gamma=@zeros([dim_p_pe]f32);
const mixed_rms=@import_module("rms_f32_local.csl",.{.rows=seq_len_p_pe,.features=dim_p_pe,.global_features=dim_p_pe*P,.epsilon=@as(f32,0.000001)});
const mixed_v_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->mixed_v[i]});
const mixed_a_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->mixed_a[i]});
const mixed_z_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->mixed_z[i]});
const mixed_work_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->mixed_work0[i]});
const mixed_projection_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->mixed_projection[i]});
const mixed_probability_snapshot_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->mixed_probability_snapshot[i]});
fn mixed_map_half(right:f16) void {@fmacs(comp_dest_dsr_1,comp_src0_dsr_1,comp_src1_dsr_1,@as(f32,right));}
fn mixed_map_float(right:f32) void {@fmacs(comp_dest_dsr_1,comp_src0_dsr_1,comp_src1_dsr_1,right);}
fn mixed_compute() void {
 swap_ptr=ptr_left_matrix_send;ptr_left_matrix_send=ptr_left_matrix_recv;ptr_left_matrix_recv=swap_ptr;
 swap_ptr=ptr_right_matrix_send;ptr_right_matrix_send=ptr_right_matrix_recv;ptr_right_matrix_recv=swap_ptr;
 if(step<P){
  comm_mod.mixed_mm(ptr_left_matrix_send,ptr_right_matrix_send,ptr_left_matrix_recv,ptr_right_matrix_recv,step,Mt,Kt,Nt,phase!=8,phase==4);
  var lh=@get_dsd(mem1d_dsd,.{.base_address=ptr_left_matrix_send,.extent=@as(u16,Mt)});
  var lf=@get_dsd(mem1d_dsd,.{.base_address=@ptrcast([*]f32,ptr_left_matrix_send),.extent=@as(u16,Mt)});
  const temp=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe}->mixed_left_column[i]});
  var rh=@get_dsd(mem1d_dsd,.{.base_address=ptr_right_matrix_send,.extent=@as(u16,Nt)});
  var rf=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{dim_p_pe}->mixed_work0[i*seq_len_p_pe]});rf=@set_dsd_base_addr(rf,@ptrcast([*]f32,ptr_right_matrix_send));
  const od=@get_dsd(mem1d_dsd,.{.base_address=@ptrcast([*]f32,ptr_out_matrix),.extent=@as(u16,Mt)});
  for(@range(i16,Kt)) |k| {
   if(phase==8){wide_ops.convert(f32,f16,temp,lh,.{});}
   @load_to_dsr(comp_dest_dsr_1,od,.{.save_address=true});@load_to_dsr(comp_src0_dsr_1,od,.{.save_address=true});
   if(phase==8){@load_to_dsr(comp_src1_dsr_1,temp,.{.save_address=false});}
   else{@load_to_dsr(comp_src1_dsr_1,lf,.{.save_address=false});}
   if(phase==4){@map(mixed_map_float,rf);rf=@increment_dsd_offset(rf,1,f32);}
   else{@map(mixed_map_half,rh);rh=@increment_dsd_offset(rh,Nt,f16);}
   lh=@increment_dsd_offset(lh,Mt,f16);lf=@increment_dsd_offset(lf,Mt,f32);
  }
  if(phase==8){input_prefix_progress[4]+=1;}else if(phase==4){attention_progress[2]+=1;}else{prelude_progress[1]+=1;}
  step+=1;@activate(next_step_id);
 }else{step=0;phase_finish();}
}
fn mixed_v_setup() void {
 Mt=seq_len_p_pe;Kt=dim_p_pe;Nt=dim_p_pe;ptr_out_matrix=@ptrcast([*]f16,&mixed_v);@fmovs(mixed_v_view,0.0);
 @fmovh(@set_dsd_base_addr(q_weight_view,@ptrcast([*]f16,&ww)),@set_dsd_base_addr(q_weight_view,@ptrcast([*]f16,&v_weight)));
 ptr_right_matrix_send=&wr;ptr_right_matrix_recv=&ww;
 swap_ptr=ptr_left_matrix_send;ptr_left_matrix_send=ptr_left_matrix_recv;ptr_left_matrix_recv=swap_ptr;matmul_compute();
}
fn mixed_o_setup() void {
 Mt=seq_len_p_pe;Kt=dim_p_pe;Nt=dim_p_pe;ptr_out_matrix=@ptrcast([*]f16,&mixed_z);@fmovs(mixed_z_view,0.0);
 @fmovh(@set_dsd_length(@set_dsd_base_addr(wv,@ptrcast([*]f16,&ww)),@as(u16,OW)),o_view);
 ptr_right_matrix_send=&wr;ptr_right_matrix_recv=&ww;
 ptr_left_matrix_send=@ptrcast([*]f16,&mixed_work1);ptr_left_matrix_recv=@ptrcast([*]f16,&mixed_work0);
 left_dim_length=L;in_preshift=true;pre_remaining=offset_step;shift_round=0;left_matrix_shift_callback();
}
fn mixed_finish() void {
 if(phase==4){wide_ops.convert(f16,f32,attention_snapshot_view,mixed_a_view,.{});attention_progress[3]+=1;@fmovs(mixed_work_view,mixed_a_view);phase=3;step=0;setup_projection();return;}
 @fmovs(mixed_projection_view,mixed_z_view);wide_ops.convert(f16,f32,projection_snapshot_view,mixed_z_view,.{});
 wide_ops.convert(f32,f16,mixed_work_view,residual_view,.{});@fadds(mixed_z_view,mixed_z_view,mixed_work_view);
 wide_ops.convert(f16,f32,z_view,mixed_z_view,.{});prelude_progress[2]=1;prelude_progress[3]+=1;
 const gh=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{dim_p_pe}->gamma[i]});const gf=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{dim_p_pe}->mixed_gamma[i]});wide_ops.convert(f32,f16,gf,gh,.{});
 mixed_rms.square_sum(&mixed_z,&mixed_work1,&mixed_rows);comm_mod.mv_allreduce_add_x_f32(&mixed_rows);mixed_rms.inverse(&mixed_rows);mixed_rms.normalize(&mixed_z,&mixed_gamma,&mixed_normalized,&mixed_rows);
 const norm=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->mixed_normalized[i]});
 wide_ops.convert(f16,f32,@set_dsd_base_addr(lv,@ptrcast([*]f16,&xwork)),norm,.{});
 @fmovh(@set_dsd_base_addr(lv,@ptrcast([*]f16,&normalized_snapshot)),@set_dsd_base_addr(lv,@ptrcast([*]f16,&xwork)));
 rms_progress[0]=1;phase=0;step=0;setup_projection();
}
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    prepare(True, "all" if args.all else 3, adapt)
