"""Resident score/softmax/value continuation within the shared projection engine."""

from binary16 import bits
from csl_region_hooks import DEFAULTS


def hooks(s, base):
    out = dict(base)
    out["ENTRY"] = (
        base["ENTRY"]
        + "phase=5;score_progress[0]=0;score_progress[1]=0;for(@range(i16,3)) |i| {attention_progress[i]=0;}attention_align=false;"
    )
    out["NEXT_BODY"] = "if(phase==5){attention_score_compute();return;}"
    out["RIGHT_FINISH_BODY"] = "if(attention_align){attention_value_align();return;}"
    out["RIGHT_INCREMENT"] = (
        "if(phase==4){right_matrix_dsd=@increment_dsd_offset(right_matrix_dsd,1,f16);}else{"
        + DEFAULTS["RIGHT_INCREMENT"]
        + "}"
    )
    out["PRESHIFT_RECORD"] = (
        "if(phase==4){attention_progress[1]=@as(u16,shift_round);}else{"
        + base["PRESHIFT_RECORD"]
        + "}"
    )
    out["STEP_RECORD"] = (
        "if(phase==4){attention_progress[2]+=1;}else{" + base["STEP_RECORD"] + "}"
    )
    out["SETUP_BODY"] = (
        "if(phase==5){attention_score_setup();return;}if(phase==4){attention_value_setup();return;}"
        + base["SETUP_BODY"]
    )
    out["OBSERVE_BODY"] = """
 if(phase==4){
  if(sampled!=0){
   @fmovh(@increment_dsd_offset(attention_value_history_view,step*L,f16),z_view);
   @fmovh(@increment_dsd_offset(attention_value_left_view,step*S,f16),@set_dsd_base_addr(attention_score_view,ptr_left_matrix_send));
   @fmovh(@increment_dsd_offset(attention_value_right_view,step*L,f16),@set_dsd_base_addr(lv,ptr_right_matrix_send));
  }
  return;
 }
""" + base["OBSERVE_BODY"]
    out["PHASE_FINISH_BODY"] = """
 if(phase==5){
  score_progress[2]+=1;@fmovh(attention_logits_view,attention_score_view);
  attention_softmax();@fmovh(attention_probability_view,attention_score_view);
  if(sampled!=0){@fmovh(attention_exponents_view,attention_partial_view);}
  phase=4;step=0;setup_projection();return;
 }
 if(phase==4){
  @fmovh(attention_snapshot_view,z_view);attention_progress[3]+=1;
  @fmovh(@set_dsd_base_addr(lv,@ptrcast([*]f16,&xwork)),z_view);
  phase=3;step=0;setup_projection();return;
 }
""" + base["PHASE_FINISH_BODY"]
    out["DECLARATIONS"] += (
        f'\nconst attention_scale:f16=@bitcast(f16,@as(u16,{bits(s["scale"])}));\n'
        + DECLARATIONS
    )
    return out


DECLARATIONS = """
const S:i16=seq_len_p_pe*seq_len_p_pe;
const attention_math=@import_module("<math>");
const attention_softmax_local=@import_module("softmax_local.csl",.{.rows=seq_len_p_pe,.cols=seq_len_p_pe});
var attention_k=@zeros([L]f16);var attention_v=@zeros([L]f16);var attention_snapshot=@zeros([L]f16);
var attention_logits=@zeros([S]f16);var attention_probability=@zeros([S]f16);
var attention_peaks=@zeros([seq_len_p_pe]f16);var attention_sums=@zeros([seq_len_p_pe]f16);
var score_history=@zeros([if(sampled!=0) P*S else 1]f16);var score_owners=@zeros([if(sampled!=0) P*L else 1]f16);
var score_roots=@zeros([P]u16);var score_progress=@zeros([3]u16);var attention_progress=@zeros([4]u16);var attention_softmax_progress=@zeros([6]u16);
var attention_value_history=@zeros([if(sampled!=0) P*L else 1]f16);var attention_value_left=@zeros([if(sampled!=0) P*S else 1]f16);var attention_value_right=@zeros([if(sampled!=0) P*L else 1]f16);
var attention_softmax_history=@zeros([if(sampled!=0) 5*seq_len_p_pe else 1]f16);var attention_exponents=@zeros([if(sampled!=0) S else 1]f16);
const attention_key_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->attention_k[i]});const attention_value_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->attention_v[i]});
const attention_score_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->up[i]});const attention_partial_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->gate[i]});
const attention_snapshot_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->attention_snapshot[i]});const attention_logits_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->attention_logits[i]});const attention_probability_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->attention_probability[i]});
const attention_exponents_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->attention_exponents[i]});
const score_history_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->score_history[i]});const score_owners_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->score_owners[i]});
const attention_value_history_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->attention_value_history[i]});const attention_value_left_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->attention_value_left[i]});const attention_value_right_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->attention_value_right[i]});
const attention_softmax_history_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe}->attention_softmax_history[i]});
var attention_align:bool=false;var attention_remaining:i16=0;var attention_align_step:i16=0;
fn attention_score_reduce() void {
 var offset=@as(i16,attention_math.floor_f16(@as(f16,offset_step+step-1)/@as(f16,P)));
 offset=offset_step+step-1-offset*P;
 const root:i16=if(offset==0) 0 else if(offset<=P/2) 2*offset-1 else 2*(P-offset);
 comm_mod.matmul_T_reduce_add_x(root,&gate,&up);
 score_roots[step-1]=@as(u16,root);
 if(@as(i16,layout_mod.get_x_coord())==root){score_progress[1]+=1;}
 @fmovh(attention_partial_view,0.0);
}
fn attention_score_compute() void {
 swap_ptr=ptr_right_matrix_send;ptr_right_matrix_send=ptr_right_matrix_recv;ptr_right_matrix_recv=swap_ptr;
 if(step<P){
  @unblock(two_hop_comm_finish_id);comm_mod.mm_two_hop_comm_T(ptr_right_matrix_send,ptr_right_matrix_recv,step);
  left_matrix_dsd=@set_dsd_base_addr(left_matrix_dsd,ptr_left_matrix_send);right_matrix_dsd=@set_dsd_base_addr(right_matrix_dsd,ptr_right_matrix_send);
  for(@range(i16,Kt)) |k| {
   out_matrix_dsd=@set_dsd_base_addr(out_matrix_dsd,ptr_out_matrix);
   @load_to_dsr(comp_dest_dsr_1,out_matrix_dsd,.{.save_address=true});@load_to_dsr(comp_src0_dsr_1,out_matrix_dsd,.{.save_address=true});@load_to_dsr(comp_src1_dsr_1,left_matrix_dsd,.{.save_address=false});@map(matmul_map_func,right_matrix_dsd);
   left_matrix_dsd=@increment_dsd_offset(left_matrix_dsd,Mt,f16);right_matrix_dsd=@increment_dsd_offset(right_matrix_dsd,Nt,f16);
  }
  if(sampled!=0){@fmovh(@increment_dsd_offset(score_history_view,step*S,f16),attention_partial_view);@fmovh(@increment_dsd_offset(score_owners_view,step*L,f16),@set_dsd_base_addr(lv,ptr_right_matrix_send));}
  score_progress[0]+=1;step+=1;attention_score_reduce();@activate(next_step_id);
 }else{step=0;phase_finish();}
}
fn attention_score_setup() void {
 @fmovh(@set_dsd_base_addr(lv,@ptrcast([*]f16,&xwork)),attention_key_view);@fmovh(attention_partial_view,0.0);
 ptr_left_matrix_send=&x;ptr_right_matrix_send=&xrecv;ptr_right_matrix_recv=&xwork;ptr_out_matrix=&gate;
 Mt=seq_len_p_pe;Kt=dim_p_pe;Nt=seq_len_p_pe;
 left_matrix_dsd=@set_dsd_length(left_matrix_dsd,@as(u16,Mt));out_matrix_dsd=@set_dsd_length(out_matrix_dsd,@as(u16,Mt));
 right_matrix_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->dummy[i]});right_matrix_dsd=@set_dsd_length(right_matrix_dsd,@as(u16,Nt));
 in_preshift=false;step=0;attention_score_compute();
}
fn attention_softmax_observe(index:i16,values:[*]f16) void {
 if(sampled!=0){@fmovh(@increment_dsd_offset(attention_softmax_history_view,index*seq_len_p_pe,f16),@set_dsd_base_addr(attention_softmax_history_view,values));}
 attention_softmax_progress[index]=1;
}
fn attention_softmax() void {
 for(@range(i16,6)) |i| {attention_softmax_progress[i]=0;}
 attention_softmax_local.scale_max(&up,&attention_peaks,attention_scale);attention_softmax_observe(0,&attention_peaks);
 comm_mod.mv_allreduce_max_x(&attention_peaks);attention_softmax_observe(1,&attention_peaks);
 attention_softmax_local.shift_exp(&up,&attention_peaks,&gate);
 attention_softmax_local.local_sum(&gate,&attention_sums);attention_softmax_observe(2,&attention_sums);
 comm_mod.mv_allreduce_add_x(&attention_sums);attention_softmax_observe(3,&attention_sums);
 attention_softmax_local.normalize(&up,&gate,&attention_sums);attention_softmax_observe(4,&attention_sums);attention_softmax_progress[5]=1;
}
fn attention_value_setup() void {
 @fmovh(@set_dsd_base_addr(lv,@ptrcast([*]f16,&xwork)),attention_value_view);@fmovh(z_view,0.0);
 ptr_left_matrix_send=&gate;ptr_left_matrix_recv=&up;ptr_right_matrix_send=&xrecv;ptr_right_matrix_recv=&xwork;ptr_out_matrix=&post_projection_z;
 Mt=seq_len_p_pe;Kt=seq_len_p_pe;Nt=dim_p_pe;left_dim_length=S;
 left_matrix_dsd=@set_dsd_length(left_matrix_dsd,@as(u16,Mt));out_matrix_dsd=@set_dsd_length(out_matrix_dsd,@as(u16,Mt));
 right_matrix_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{dim_p_pe}->dummy[i*seq_len_p_pe]});
 const px=@as(i16,layout_mod.get_x_coord());attention_remaining=if(px==0) 0 else if(px%2==0) P-px/2 else (px+1)/2;
 attention_align_step=0;attention_align=true;attention_value_align();
}
fn attention_value_align() void {
 if(attention_remaining>0){attention_remaining-=1;swap_ptr=ptr_right_matrix_send;ptr_right_matrix_send=ptr_right_matrix_recv;ptr_right_matrix_recv=swap_ptr;comm_mod.mm_two_hop_comm_T(ptr_right_matrix_send,ptr_right_matrix_recv,attention_align_step);attention_align_step+=1;}
 else{attention_align=false;attention_progress[0]=@as(u16,attention_align_step);in_preshift=true;pre_remaining=offset_step;shift_round=0;left_matrix_shift_callback();}
}
"""


EXPORTS = {
    "attention_k": "f16",
    "attention_v": "f16",
    "attention_snapshot": "f16",
    "attention_logits": "f16",
    "attention_probability": "f16",
    "score_history": "f16",
    "score_owners": "f16",
    "attention_value_history": "f16",
    "attention_value_left": "f16",
    "attention_value_right": "f16",
    "attention_softmax_history": "f16",
    "attention_exponents": "f16",
    "score_roots": "u16",
    "score_progress": "u16",
    "attention_progress": "u16",
    "attention_softmax_progress": "u16",
}
DECLARATIONS += "\n" + "\n".join(
    f'var {name}_ptr:[*]{dtype}=&{name};\ncomptime {{@export_symbol({name}_ptr,"{name}");}}'
    for name, dtype in EXPORTS.items()
)
