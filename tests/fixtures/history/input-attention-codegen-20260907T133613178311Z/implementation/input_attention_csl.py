"""Joined input RMS/QKV/pair prefix over the shared resident projection engine."""


def hooks(s, base):
    out = dict(base)
    out["ENTRY"] = base["ENTRY"] + """
 for(@range(i16,12)) |i| {if(i!=11){input_prefix_progress[i]=0;}}
 rms_local.square_sum(&residual,&xrecv,&rms_rows);
 comm_mod.mv_allreduce_add_x(&rms_rows);rms_local.inverse(&rms_rows);
 rms_local.normalize(&residual,&gamma,&xwork,&rms_rows);
 @fmovh(input_normalized_view,@set_dsd_base_addr(lv,@ptrcast([*]f16,&xwork)));
 input_prefix_progress[0]=1;phase=6;
"""
    out["SETUP_BODY"] = (
        "if(phase>=6){input_projection_setup();return;}" + base["SETUP_BODY"]
    )
    out["PRESHIFT_RECORD"] = (
        "if(phase>=6){input_prefix_progress[1]=@as(u16,shift_round);}else{"
        + base["PRESHIFT_RECORD"]
        + "}"
    )
    out["STEP_RECORD"] = (
        "if(phase>=6){input_prefix_progress[2+phase-6]+=1;}else{"
        + base["STEP_RECORD"]
        + "}"
    )
    out["OBSERVE_BODY"] = """
 if(phase>=6){
  if(sampled!=0){
   const branch:i16=phase-6;
   @fmovh(@increment_dsd_offset(input_projection_history_view,(branch*P+step)*L,f16),@set_dsd_base_addr(lv,ptr_out_matrix));
   if(step==0){
    @fmovh(@increment_dsd_offset(input_left_first_view,branch*L,f16),@set_dsd_base_addr(lv,ptr_left_matrix_send));
    @fmovh(@increment_dsd_offset(input_right_first_view,branch*OW,f16),@set_dsd_base_addr(q_weight_view,ptr_right_matrix_send));
   }
  }
  return;
 }
""" + base["OBSERVE_BODY"]
    out["PHASE_FINISH_BODY"] = """
 if(phase>=6){
  input_prefix_progress[5+phase-6]=1;
  if(phase==6){@fmovh(input_q_raw_view,lv);phase=7;setup_projection();return;}
  if(phase==7){@fmovh(input_k_raw_view,attention_key_view);phase=8;setup_projection();return;}
  input_pair.apply(&x,&x,&cosine,&sine,&xwork,&input_pair_history);
  input_prefix_progress[8]=1;
  const pair_history_k:[*]f16=if(sampled!=0) @ptrcast([*]f16,&input_pair_history)+2*L else &input_pair_history;
  input_pair.apply(&attention_k,&attention_k,&cosine,&sine,&xwork,pair_history_k);
  input_prefix_progress[9]=1;input_prefix_progress[10]=1;input_prefix_progress[11]+=1;
  phase=5;step=0;setup_projection();return;
 }
""" + base["PHASE_FINISH_BODY"]
    out["DECLARATIONS"] += DECLARATIONS
    return out


DECLARATIONS = """
const input_pair=@import_module("pair_rotation_local.csl",.{.rows=seq_len_p_pe,.features=dim_p_pe,.sampled=sampled,.swapped=1});
var q_weight=@zeros([OW]f16);var k_weight=@zeros([OW]f16);var v_weight=@zeros([OW]f16);
var cosine=@zeros([dim_p_pe/2]f16);var sine=@zeros([dim_p_pe/2]f16);
var input_normalized=@zeros([L]f16);var input_q_raw=@zeros([L]f16);var input_k_raw=@zeros([L]f16);
var input_projection_history=@zeros([if(sampled!=0) 3*P*L else 1]f16);
var input_left_first=@zeros([if(sampled!=0) 3*L else 1]f16);
var input_right_first=@zeros([if(sampled!=0) 3*OW else 1]f16);
var input_pair_history=@zeros([if(sampled!=0) 4*L else 1]f16);
var input_prefix_progress=@zeros([12]u16);
const q_weight_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{OW}->q_weight[i]});
const input_normalized_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->input_normalized[i]});
const input_q_raw_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->input_q_raw[i]});
const input_k_raw_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->input_k_raw[i]});
const input_projection_history_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->input_projection_history[i]});
const input_left_first_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->input_left_first[i]});
const input_right_first_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{OW}->input_right_first[i]});
fn input_projection_setup() void {
 Mt=seq_len_p_pe;Kt=dim_p_pe;Nt=dim_p_pe;
 var weight:[*]f16=&q_weight;ptr_out_matrix=&x;
 if(phase==7){weight=&k_weight;ptr_out_matrix=&attention_k;}
 else if(phase==8){weight=&v_weight;ptr_out_matrix=&attention_v;}
 @fmovh(@set_dsd_base_addr(lv,ptr_out_matrix),0.0);
 @fmovh(@set_dsd_base_addr(q_weight_view,@ptrcast([*]f16,&ww)),@set_dsd_base_addr(q_weight_view,weight));
 ptr_right_matrix_send=&wr;ptr_right_matrix_recv=&ww;
 left_matrix_dsd=@set_dsd_length(left_matrix_dsd,@as(u16,Mt));
 out_matrix_dsd=@set_dsd_length(out_matrix_dsd,@as(u16,Mt));
 right_matrix_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->dummy[i]});
 right_matrix_dsd=@set_dsd_length(right_matrix_dsd,@as(u16,Nt));
 if(phase==6){
  ptr_left_matrix_send=&xrecv;ptr_left_matrix_recv=&xwork;
  left_dim_length=L;in_preshift=true;pre_remaining=offset_step;shift_round=0;
  left_matrix_shift_callback();
 }else{
  swap_ptr=ptr_left_matrix_send;ptr_left_matrix_send=ptr_left_matrix_recv;ptr_left_matrix_recv=swap_ptr;
  matmul_compute();
 }
}
"""

EXPORTS = {
    name: "f16"
    for name in (
        "q_weight",
        "k_weight",
        "v_weight",
        "cosine",
        "sine",
        "input_normalized",
        "input_q_raw",
        "input_k_raw",
        "input_projection_history",
        "input_left_first",
        "input_right_first",
        "input_pair_history",
    )
}
EXPORTS["input_prefix_progress"] = "u16"
DECLARATIONS += "\n" + "\n".join(
    f'var {name}_ptr:[*]{dtype}=&{name};\ncomptime {{@export_symbol({name}_ptr,"{name}");}}'
    for name, dtype in EXPORTS.items()
)
