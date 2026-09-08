"""One-time source authoring for device-aligned, strided-right matrix contraction."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


from pathlib import Path

ROOT = repository_root(__file__)
rt = ROOT / "runtime/csl"
assert not (rt / "device_matmul_pe.csl").exists()
source = (
    ROOT / "validation/evidence/attention-value-source-20260907T022427483584Z/prefill.csl"
).read_text()


def fn(name):
    a = source.index("fn " + name + "(")
    i = source.index("{", a) + 1
    depth = 1
    while depth:
        depth += (source[i] == "{") - (source[i] == "}")
        i += 1
    return source[a:i] + "\n"


pe = """// Source-derived device-aligned matmul: MeshInfra/WaferLLM fd1c2daae37cd68706c03fc8009887ecee9900f8, Apache-2.0.
param memcpy_params;param comm_params;param P:i16;param dim_p_pe:i16;param seq_len_p_pe:i16;param sampled:i16;
const S:i16=seq_len_p_pe*seq_len_p_pe;const L:i16=seq_len_p_pe*dim_p_pe;
const left_matrix_finish_id=@get_local_task_id(19);const right_matrix_finish_id=@get_local_task_id(20);const next_step_id=@get_local_task_id(26);const two_hop_comm_finish_id=@get_local_task_id(25);
const sys_mod=@import_module("<memcpy/memcpy>",memcpy_params);const layout_mod=@import_module("<layout>");const timestamp=@import_module("<time>");const config=@import_module("<tile_config>");
const comm_mod=@import_module("inference_comm.csl",.{.comm_params=comm_params,.P=P,.dim_p_pe=dim_p_pe,.seq_len_p_pe=seq_len_p_pe,.ffn_dim_p_pe=dim_p_pe,.left_matrix_finish_id=left_matrix_finish_id,.right_matrix_finish_id=right_matrix_finish_id});
var A_input=@zeros([S]f16);var B_input=@zeros([L]f16);var score=@zeros([S]f16);var XV_tile=@zeros([L]f16);var seqLen_seqLen_tmp=@zeros([S]f16);var seqLen_dim_tmp=@zeros([L]f16);var output_tile=@zeros([L]f16);
const ad=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->A_input[i]});const bd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->B_input[i]});const aw=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->score[i]});const bw=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->XV_tile[i]});
const output_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->output_tile[i]});const dummy=@zeros([1]f16);
var left_matrix_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->dummy[i]});var right_matrix_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->dummy[i]});var out_matrix_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->dummy[i]});
const comp_dest_dsr_1=@get_dsr(dsr_dest,1);const comp_src0_dsr_1=@get_dsr(dsr_src0,1);const comp_src1_dsr_1=@get_dsr(dsr_src1,1);
var ptr_left_matrix_send:[*]f16;var ptr_left_matrix_recv:[*]f16;var ptr_right_matrix_send:[*]f16;var ptr_right_matrix_recv:[*]f16;var ptr_out_matrix:[*]f16;var swap_ptr:[*]f16;
var Mt:i16=0;var Nt:i16=0;var Kt:i16=0;var step:i16=0;var offset_step:i16=0;var px:i16=0;var py:i16=0;var left_dim_length:i16=0;var in_preshift:bool=false;var pre_remaining:i16=0;var shift_round:i16=0;var is_T_matmul:bool=false;
var value_phase:bool=false;var value_remaining:i16=0;var value_step:i16=0;
var history=@zeros([if(sampled!=0) P*L else 1]f16);var left_owners=@zeros([if(sampled!=0) P*S else 1]f16);var right_owners=@zeros([if(sampled!=0) P*L else 1]f16);
const hd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->history[i]});const ld=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->left_owners[i]});const rd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->right_owners[i]});
var progress=@zeros([4]u16);var timing=@zeros([6]u16);var queues=@zeros([2]u16);var start=@zeros([3]u16);var end=@zeros([3]u16);
fn init_task() void {px=@as(i16,layout_mod.get_x_coord());py=@as(i16,layout_mod.get_y_coord());if(py==0){offset_step=0;}else if(py%2==0){offset_step=P-py/2;}else{offset_step=(py+1)/2;}comm_mod.init_(px,py);sys_mod.unblock_cmd_stream();}
task left_matrix_finish() void {@block(left_matrix_finish_id);if(in_preshift){left_matrix_shift_callback();}else{@unblock(two_hop_comm_finish_id);}}
task right_matrix_finish() void {@block(right_matrix_finish_id);if(value_phase){value_shift();return;}@activate(two_hop_comm_finish_id);}
task two_hop_comm_finish() void {@block(two_hop_comm_finish_id);@unblock(next_step_id);}
task next_step() void {@block(next_step_id);matmul_compute();}
"""
pe += fn("left_matrix_shift_callback").replace(
    "        step = 0;", "        progress[1]=@as(u16,shift_round);step = 0;"
)
pe += fn("matmul_map_func")
body = fn("matmul_compute").replace("prefill_struct();", "hls_finish();")
needle = "        @fmovh(@increment_dsd_offset(hd,step*seq_len_p_pe*dim_p_pe,f16),output_dsd);"
assert body.count(needle) == 1
body = body.replace(
    needle,
    """        if(sampled!=0){@fmovh(@increment_dsd_offset(hd,step*L,f16),output_dsd);@fmovh(@increment_dsd_offset(ld,step*S,f16),@set_dsd_base_addr(aw,ptr_left_matrix_send));@fmovh(@increment_dsd_offset(rd,step*L,f16),@set_dsd_base_addr(bw,ptr_right_matrix_send));}
        progress[2]+=1;""",
)
pe += (
    body
    + fn("output_matmul")
    + fn("value_shift").replace(
        "value_phase=false;", "value_phase=false;progress[0]=@as(u16,value_step);"
    )
)
pe += """
fn hls_main() void {timestamp.enable_tsc();timestamp.get_timestamp(&start);for(@range(i16,3)) |i| {progress[i]=0;}@fmovh(aw,ad);@fmovh(bw,bd);output_matmul();}
fn hls_finish() void {progress[3]+=1;timestamp.get_timestamp(&end);timestamp.disable_tsc();for(@range(i16,3)) |i| {timing[i]=start[i];timing[i+3]=end[i];}const qi=config.input_queue_status.get();const qo=config.output_queue_status.get();queues[0]=@as(u16,qi.empty);queues[1]=@as(u16,qo.empty);sys_mod.unblock_cmd_stream();}
var ap:[*]f16=&A_input;var bp:[*]f16=&B_input;var op:[*]f16=&output_tile;var hp:[*]f16=&history;var lp:[*]f16=&left_owners;var rp:[*]f16=&right_owners;var pp:[*]u16=&progress;var tp:[*]u16=&timing;var qp:[*]u16=&queues;
comptime {@bind_local_task(left_matrix_finish,left_matrix_finish_id);@block(left_matrix_finish_id);@bind_local_task(right_matrix_finish,right_matrix_finish_id);@block(right_matrix_finish_id);@bind_local_task(two_hop_comm_finish,two_hop_comm_finish_id);@block(two_hop_comm_finish_id);@bind_local_task(next_step,next_step_id);@block(next_step_id);
@export_symbol(ap,"a");@export_symbol(bp,"b");@export_symbol(op,"result");@export_symbol(hp,"history");@export_symbol(lp,"left_owners");@export_symbol(rp,"right_owners");@export_symbol(pp,"progress");@export_symbol(tp,"timing");@export_symbol(qp,"queues");@export_symbol(init_task);@export_symbol(hls_main);}
"""
(rt / "device_matmul_pe.csl").write_text(pe)
layout = (rt / "score_layout.csl").read_text()
import re

layout = re.sub(r"@export_name\([^;]*?\);", "", layout)
pos = layout.rfind("}")
layout = (
    layout[:pos]
    + """@export_name("a",[*]f16,true);@export_name("b",[*]f16,true);@export_name("result",[*]f16,true);@export_name("history",[*]f16,true);@export_name("left_owners",[*]f16,true);@export_name("right_owners",[*]f16,true);@export_name("progress",[*]u16,true);@export_name("timing",[*]u16,true);@export_name("queues",[*]u16,true);@export_name("init_task",fn()void);@export_name("hls_main",fn()void);
"""
    + layout[pos:]
)
(rt / "device_matmul_layout.csl").write_text(layout)
