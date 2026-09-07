"""One-time source authoring; never executed by the compiler."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
rt = ROOT / "toolchain/runtime"
src = ROOT / "projects/waferllm/upstream/Prefill/src/prefill.csl"
assert not (rt / "score_pe.csl").exists()
original = src.read_text()


def fn(name):
    start = original.index("fn " + name + "(")
    left = original.index("{", start)
    level = 1
    i = left + 1
    while level:
        level += (original[i] == "{") - (original[i] == "}")
        i += 1
    return original[start:i] + "\n"


pe = """// Source-derived QK-transpose schedule; MeshInfra/WaferLLM fd1c2daae37cd68706c03fc8009887ecee9900f8, Apache-2.0.
param memcpy_params;param comm_params;param P:i16;param dim_p_pe:i16;param seq_len_p_pe:i16;param sampled:i16;
const L:i16=seq_len_p_pe*dim_p_pe;const S:i16=seq_len_p_pe*seq_len_p_pe;
const left_matrix_finish_id=@get_local_task_id(19);const right_matrix_finish_id=@get_local_task_id(20);
const next_step_id=@get_local_task_id(26);const two_hop_comm_finish_id=@get_local_task_id(25);
const sys_mod=@import_module("<memcpy/memcpy>",memcpy_params);const math_lib=@import_module("<math>");const layout_mod=@import_module("<layout>");const timestamp=@import_module("<time>");const config=@import_module("<tile_config>");
const comm_mod=@import_module("inference_comm.csl",.{.comm_params=comm_params,.P=P,.dim_p_pe=dim_p_pe,.seq_len_p_pe=seq_len_p_pe,.ffn_dim_p_pe=dim_p_pe,.left_matrix_finish_id=left_matrix_finish_id,.right_matrix_finish_id=right_matrix_finish_id});
var XQ_tile=@zeros([L]f16);var K_input=@zeros([L]f16);var XK_tile=@zeros([L]f16);var seqLen_dim_tmp=@zeros([L]f16);
var score=@zeros([S]f16);var seqLen_seqLen_tmp=@zeros([S]f16);
var ptr_XQ:[*]f16=&XQ_tile;var ptr_K:[*]f16=&K_input;var ptr_score:[*]f16=&score;var ptr_seqLen_seqLen_tmp:[*]f16=&seqLen_seqLen_tmp;
const ki=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->K_input[i]});const kw=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->XK_tile[i]});
const seqLen_seqLen_tmp_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->seqLen_seqLen_tmp[i]});
const dummy=@zeros([1]f16);
var left_matrix_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->dummy[i]});var right_matrix_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->dummy[i]});var out_matrix_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->dummy[i]});
const comp_dest_dsr_1=@get_dsr(dsr_dest,1);const comp_src0_dsr_1=@get_dsr(dsr_src0,1);const comp_src1_dsr_1=@get_dsr(dsr_src1,1);
var ptr_left_matrix_send:[*]f16;var ptr_right_matrix_send:[*]f16;var ptr_right_matrix_recv:[*]f16;var ptr_out_matrix:[*]f16;var swap_ptr:[*]f16;
var Mt:i16=0;var Nt:i16=0;var Kt:i16=0;var step:i16=0;var offset_step:i16=0;var current_offset:i16=0;var root:i16=P/2;var reduce_root:i16=0;var px:i16=0;var py:i16=0;var is_T_matmul:bool=true;
var history=@zeros([if(sampled!=0) P*S else 1]f16);var owners=@zeros([if(sampled!=0) P*L else 1]f16);var roots=@zeros([P]u16);
const hd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->history[i]});const od=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->owners[i]});
var progress=@zeros([3]u16);var timing=@zeros([6]u16);var queues=@zeros([2]u16);var start=@zeros([3]u16);var end=@zeros([3]u16);
fn init_task() void {px=@as(i16,layout_mod.get_x_coord());py=@as(i16,layout_mod.get_y_coord());if(py==0){offset_step=0;}else if(py%2==0){offset_step=P-py/2;}else{offset_step=(py+1)/2;}comm_mod.init_(px,py);sys_mod.unblock_cmd_stream();}
task left_matrix_finish() void {@block(left_matrix_finish_id);@unblock(two_hop_comm_finish_id);}
task right_matrix_finish() void {@block(right_matrix_finish_id);@activate(two_hop_comm_finish_id);}
task two_hop_comm_finish() void {@block(two_hop_comm_finish_id);@unblock(next_step_id);}
task next_step() void {@block(next_step_id);matmul_T_compute();}
"""
pe += fn("matmul_map_func")
reduce = fn("matmul_T_reduce_add")
needle = (
    "    comm_mod.matmul_T_reduce_add_x(reduce_root, ptr_seqLen_seqLen_tmp, ptr_score);"
)
assert needle in reduce
reduce = reduce.replace(
    needle,
    needle
    + "\n    roots[step-1]=@as(u16,reduce_root);if(px==reduce_root){progress[1]+=1;}",
)
pe += reduce
compute = fn("matmul_T_compute")
needle = "        step += 1;"
assert needle in compute
compute = compute.replace(
    needle,
    """        if(sampled!=0){@fmovh(@increment_dsd_offset(hd,step*S,f16),seqLen_seqLen_tmp_dsd);@fmovh(@increment_dsd_offset(od,step*L,f16),@set_dsd_base_addr(kw,ptr_right_matrix_send));}
        progress[0]+=1;
"""
    + needle,
).replace("prefill_struct();", "hls_finish();")
pe += compute + fn("score_matmul")
pe += """
fn hls_main() void {timestamp.enable_tsc();timestamp.get_timestamp(&start);progress[0]=0;progress[1]=0;@fmovh(kw,ki);score_matmul();}
fn hls_finish() void {progress[2]+=1;timestamp.get_timestamp(&end);timestamp.disable_tsc();for(@range(i16,3)) |i| {timing[i]=start[i];timing[i+3]=end[i];}const qi=config.input_queue_status.get();const qo=config.output_queue_status.get();queues[0]=@as(u16,qi.empty);queues[1]=@as(u16,qo.empty);sys_mod.unblock_cmd_stream();}
var hp:[*]f16=&history;var op:[*]f16=&owners;var rp:[*]u16=&roots;var pp:[*]u16=&progress;var tp:[*]u16=&timing;var qp:[*]u16=&queues;
comptime {@bind_local_task(left_matrix_finish,left_matrix_finish_id);@block(left_matrix_finish_id);@bind_local_task(right_matrix_finish,right_matrix_finish_id);@block(right_matrix_finish_id);@bind_local_task(two_hop_comm_finish,two_hop_comm_finish_id);@block(two_hop_comm_finish_id);@bind_local_task(next_step,next_step_id);@block(next_step_id);
@export_symbol(ptr_XQ,"q");@export_symbol(ptr_K,"k");@export_symbol(ptr_score,"result");@export_symbol(hp,"history");@export_symbol(op,"owners");@export_symbol(rp,"roots");@export_symbol(pp,"progress");@export_symbol(tp,"timing");@export_symbol(qp,"queues");@export_symbol(init_task);@export_symbol(hls_main);}
"""
(rt / "score_pe.csl").write_text(pe)
layout = (rt / "normalized_fanout_layout.csl").read_text()
layout = layout.replace(
    "param sampled:i16;param projections:i16;param epsilon_bits:u16;\nconst epsilon=@bitcast(f16,epsilon_bits);",
    "param sampled:i16;",
).replace(",.projections=projections,.epsilon=epsilon", "")
layout = re.sub(r"@export_name\([^;]*?\);", "", layout)
pos = layout.rfind("}")
layout = (
    layout[:pos]
    + """@export_name("q",[*]f16,true);@export_name("k",[*]f16,true);@export_name("result",[*]f16,true);@export_name("history",[*]f16,true);@export_name("owners",[*]f16,true);@export_name("roots",[*]u16,true);@export_name("progress",[*]u16,true);@export_name("timing",[*]u16,true);@export_name("queues",[*]u16,true);@export_name("init_task",fn()void);@export_name("hls_main",fn()void);
"""
    + layout[pos:]
)
(rt / "score_layout.csl").write_text(layout)
