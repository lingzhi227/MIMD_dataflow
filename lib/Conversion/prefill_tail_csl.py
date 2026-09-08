"""A joined projection prelude using the same CSL projection engine and callbacks."""

from csl_region_hooks import DEFAULTS


def hooks(s, base):
    out = dict(base)
    norm_entry = base["ENTRY"]
    out["ENTRY"] = "phase=3;for(@range(i16,3)) |i| {prelude_progress[i]=0;}"
    out["PRESHIFT_RECORD"] = (
        "if(phase==3){prelude_progress[0]=@as(u16,shift_round);}else{"
        + DEFAULTS["PRESHIFT_RECORD"]
        + "}"
    )
    out["STEP_RECORD"] = (
        "if(phase==3){prelude_progress[1]+=1;}else{" + DEFAULTS["STEP_RECORD"] + "}"
    )
    out["SETUP_BODY"] = """
 if(phase==3){
  Mt=seq_len_p_pe;Kt=dim_p_pe;Nt=dim_p_pe;
  ptr_out_matrix=&post_projection_z;@fmovh(z_view,0.0);
  @fmovh(@set_dsd_length(@set_dsd_base_addr(wv,@ptrcast([*]f16,&ww)),@as(u16,OW)),o_view);
  ptr_right_matrix_send=&wr;ptr_right_matrix_recv=&ww;
  ptr_left_matrix_send=&xrecv;ptr_left_matrix_recv=&xwork;
  left_matrix_dsd=@set_dsd_length(left_matrix_dsd,@as(u16,Mt));
  out_matrix_dsd=@set_dsd_length(out_matrix_dsd,@as(u16,Mt));
  right_matrix_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->dummy[i]});
  right_matrix_dsd=@set_dsd_length(right_matrix_dsd,@as(u16,Nt));
  left_dim_length=L;in_preshift=true;pre_remaining=offset_step;shift_round=0;
  left_matrix_shift_callback();return;
 }
"""
    out["OBSERVE_BODY"] = """
 if(phase==3){
  if(sampled!=0){
   @fmovh(@increment_dsd_offset(projection_history_view,step*L,f16),z_view);
   if(step==0){
    @fmovh(projection_left_view,@set_dsd_base_addr(lv,ptr_left_matrix_send));
    @fmovh(projection_right_view,@set_dsd_length(@set_dsd_base_addr(wv,ptr_right_matrix_send),@as(u16,OW)));
   }
  }
  return;
 }
"""
    out["PHASE_FINISH_BODY"] = (
        """
 if(phase==3){
  @fmovh(projection_snapshot_view,z_view);
  @load_to_dsr(comp_dest_dsr_1,z_view);@load_to_dsr(comp_src0_dsr_1,z_view);@load_to_dsr(comp_src1_dsr_1,residual_view);
  @faddh(comp_dest_dsr_1,comp_src0_dsr_1,comp_src1_dsr_1);
  prelude_progress[2]=1;prelude_progress[3]+=1;
  @fmovh(@set_dsd_base_addr(lv,@ptrcast([*]f16,&xwork)),z_view);
"""
        + norm_entry
        + """
  phase=0;step=0;setup_projection();return;
 }
"""
    )
    out["DECLARATIONS"] += """
const OW:i16=dim_p_pe*dim_p_pe;
var output_weight=@zeros([OW]f16);var residual=@zeros([L]f16);var post_projection_z=@zeros([L]f16);
var projection_snapshot=@zeros([L]f16);
var projection_history=@zeros([if(sampled!=0) P*L else 1]f16);
var projection_left_first=@zeros([if(sampled!=0) L else 1]f16);
var projection_right_first=@zeros([if(sampled!=0) OW else 1]f16);
var prelude_progress=@zeros([4]u16);
const o_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{OW}->output_weight[i]});
const residual_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->residual[i]});
const z_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->post_projection_z[i]});
const projection_snapshot_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->projection_snapshot[i]});
const projection_history_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->projection_history[i]});
const projection_left_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->projection_left_first[i]});
const projection_right_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{OW}->projection_right_first[i]});
var output_weight_ptr:[*]f16=&output_weight;var residual_ptr:[*]f16=&residual;var post_projection_z_ptr:[*]f16=&post_projection_z;var projection_snapshot_ptr:[*]f16=&projection_snapshot;
var projection_history_ptr:[*]f16=&projection_history;var projection_left_first_ptr:[*]f16=&projection_left_first;var projection_right_first_ptr:[*]f16=&projection_right_first;var prelude_progress_ptr:[*]u16=&prelude_progress;
comptime {@export_symbol(output_weight_ptr,"output_weight");@export_symbol(residual_ptr,"residual");@export_symbol(post_projection_z_ptr,"post_projection_z");@export_symbol(projection_snapshot_ptr,"projection_snapshot");@export_symbol(projection_history_ptr,"projection_history");@export_symbol(projection_left_first_ptr,"projection_left_first");@export_symbol(projection_right_first_ptr,"projection_right_first");@export_symbol(prelude_progress_ptr,"prelude_progress");}
"""
    return out
