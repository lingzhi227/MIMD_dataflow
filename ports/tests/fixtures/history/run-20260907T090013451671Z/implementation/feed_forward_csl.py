"""Local entry/finish adapters around the shared, joined CSL MLP engine."""


def hooks(epsilon_bits):
    assert type(epsilon_bits) is int and 0 < epsilon_bits < 65536
    return dict(
        ENTRY="""rms_progress[0]=0;
 rms_local.square_sum(&xwork,&xrecv,&rms_rows);
 comm_mod.mv_allreduce_add_x(&rms_rows);
 rms_local.inverse(&rms_rows);
 rms_local.normalize(&xwork,&gamma,&xwork,&rms_rows);
 @fmovh(@set_dsd_base_addr(lv,@ptrcast([*]f16,&normalized_snapshot)),@set_dsd_base_addr(lv,@ptrcast([*]f16,&xwork)));
 rms_progress[0]=1;
 """,
        FINISH="""@fmovh(@set_dsd_base_addr(lv,@ptrcast([*]f16,&down_snapshot)),@set_dsd_base_addr(lv,@ptrcast([*]f16,&xwork)));
 @load_to_dsr(comp_dest_dsr_1,@set_dsd_base_addr(lv,@ptrcast([*]f16,&xwork)));
 @load_to_dsr(comp_src0_dsr_1,@set_dsd_base_addr(lv,@ptrcast([*]f16,&xwork)));
 @load_to_dsr(comp_src1_dsr_1,lv);@faddh(comp_dest_dsr_1,comp_src0_dsr_1,comp_src1_dsr_1);
 rms_progress[1]+=1;
 """,
        DECLARATIONS=f"""
const rms_local=@import_module("rms_local.csl",.{{.rows=seq_len_p_pe,.features=dim_p_pe,.global_features=dim_p_pe*P,.epsilon=@bitcast(f16,@as(u16,{epsilon_bits}))}});
var gamma=@zeros([dim_p_pe]f16);var rms_rows=@zeros([seq_len_p_pe]f16);
var normalized_snapshot=@zeros([L]f16);var down_snapshot=@zeros([L]f16);var rms_progress=@zeros([2]u16);
var gamma_ptr:[*]f16=&gamma;var normalized_ptr:[*]f16=&normalized_snapshot;var down_snapshot_ptr:[*]f16=&down_snapshot;var rms_progress_ptr:[*]u16=&rms_progress;
comptime {{@export_symbol(gamma_ptr,"gamma");@export_symbol(normalized_ptr,"normalized");@export_symbol(down_snapshot_ptr,"down_snapshot");@export_symbol(rms_progress_ptr,"rms_progress");}}
""",
    )
