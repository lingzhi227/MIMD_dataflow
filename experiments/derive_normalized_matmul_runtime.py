"""One-time source derivation, not compiler logic; review diff against pinned CSL."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

from pathlib import Path
import difflib
ROOT = repository_root(__file__)
src=ROOT/'third_party/sources/waferllm/Prefill/src'
s=(src/'prefill.csl').read_text()
def between(first,last):
    a=s.index(first);return s[a:s.index(last,a)]
header=between('param memcpy_params;', '// * X: input')
header=header.replace('param ffn_dim_p_pe: i16;','const ffn_dim_p_pe:i16=dim_p_pe;\nparam sampled:i16;\nparam epsilon:f16;').replace('const eps: f16 = 0.000001;','const eps:f16=epsilon;')
header=header.replace('    alpha = 1.0 / @as(f16, math_lib.sqrt(head_dim));','')
header=header.replace('"comm_lib/comm_pe.csl"','"inference_comm.csl"')
parts=[header,
 between('// * X: input','var K_weight_tile:'),
 between('var X_norm_tile:', 'var XK_tile:'),
 between('var local_sum:', 'var local_max:'),
 between('var swap_ptr:', 'var seqLen_seqLen_tmp:'),
 between('var dim_dim_tmp:', 'fn matmul_T_reduce_add()'),
 between('fn matmul_compute()', 'fn rmsnorm_x()'),
 between('fn rmsnorm_x()', 'fn xk_matmul()')]
text='// Derived from WaferLLM Prefill fd1c2daae37cd68706c03fc8009887ecee9900f8.\n// Resident RMSNorm -> matrix projection, with explicit row-scale correction.\n'+''.join(parts)
text=text.replace('    if (is_T_matmul) {\n        matmul_T_compute();\n    } else {\n        matmul_compute();\n    }','    matmul_compute();')
text=text.replace('        prefill_struct();','        hls_finish();')
text=text.replace('    prefill_struct();','')
needle='    for (@range(i16, dim_p_pe)) |i| {\n        @fmulh(comp_dest_dsr_1, comp_src0_dsr_1, local_sum[i]);\n    }'
assert text.count(needle)==1
text=text.replace(needle,'    @load_to_dsr(comp_src1_dsr_1, local_sum_dsd, .{ .save_address = false });\n    for (@range(i16, dim_p_pe)) |i| {\n        @fmulh(comp_dest_dsr_1, comp_src0_dsr_1, comp_src1_dsr_1);\n    }')
text=text.replace('        step += 1;','        if(sampled!=0){const hd=@increment_dsd_offset(hls_hist_dsd,step*seq_len_p_pe*dim_p_pe,f16);@fmovh(hd,XQ_dsd);}\n        hls_progress[1]+=1;\n        step += 1;')
text+='''
const config=@import_module("<tile_config>");
var hls_norm=@zeros([if(sampled!=0) seq_len_p_pe*dim_p_pe else 1]f16);
var hls_history=@zeros([if(sampled!=0) P*seq_len_p_pe*dim_p_pe else 1]f16);
const hls_norm_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*dim_p_pe}->hls_norm[i]});
const hls_hist_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*dim_p_pe}->hls_history[i]});
const normalized_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*dim_p_pe}->X_norm_tile[i]});
var hls_progress=@zeros([4]u16);var hls_timing=@zeros([6]u16);var hls_queues=@zeros([2]u16);
fn hls_main() void {
 timestamp.enable_tsc();timestamp.get_timestamp(&tscStartBuffer);
 for(@range(i16,3)) |i| {hls_progress[i]=0;}
 rmsnorm_x();hls_progress[0]=1;
 if(sampled!=0){@fmovh(hls_norm_dsd,normalized_dsd);}
 xq_matmul();
}
fn hls_finish() void {
 hls_progress[2]=1;hls_progress[3]+=1;
 timestamp.get_timestamp(&tscEndBuffer);timestamp.disable_tsc();
 for(@range(i16,3)) |i| {hls_timing[i]=tscStartBuffer[i];hls_timing[i+3]=tscEndBuffer[i];}
 const qi=config.input_queue_status.get();const qo=config.output_queue_status.get();
 hls_queues[0]=@as(u16,qi.empty);hls_queues[1]=@as(u16,qo.empty);
 sys_mod.unblock_cmd_stream();
}
var hp:[*]f16=&hls_history;var np:[*]f16=&hls_norm;var pp:[*]u16=&hls_progress;
var tp:[*]u16=&hls_timing;var qp:[*]u16=&hls_queues;
comptime {
 @bind_local_task(two_hop_comm_finish,two_hop_comm_finish_id);@block(two_hop_comm_finish_id);
 @bind_local_task(left_matrix_finish,left_matrix_finish_id);@block(left_matrix_finish_id);
 @bind_local_task(right_matrix_finish,right_matrix_finish_id);@block(right_matrix_finish_id);
 @bind_local_task(next_step,next_step_id);@block(next_step_id);
 @export_symbol(ptr_X,"X");@export_symbol(ptr_W,"W");@export_symbol(ptr_Q_weight,"Q_weight");
 @export_symbol(ptr_XQ,"result");@export_symbol(hp,"history");@export_symbol(np,"normalized");
 @export_symbol(pp,"progress");@export_symbol(tp,"timing");@export_symbol(qp,"queues");
 @export_symbol(init_task);@export_symbol(hls_main);
}
'''
runtime=ROOT/'runtime/csl'
for name,content in [('normalized_matmul_pe.csl',text),('inference_comm.csl',(src/'comm_lib/comm_pe.csl').read_text()),('inference_routes.csl',(src/'comm_lib/comm_layout.csl').read_text())]:
 p=runtime/name;assert not p.exists();p.write_text(content)
layout=(src/'layout.csl').read_text()
layout=layout.replace('param ffn_dim_p_pe: i16;','param ffn_dim_p_pe:i16;\nparam sampled:i16;param epsilon_bits:u16;\nconst epsilon=@bitcast(f16,epsilon_bits);')
layout=layout.replace('"comm_lib/comm_layout.csl"','"inference_routes.csl"').replace('"prefill.csl"','"pe.csl"')
layout=layout.replace('.ffn_dim_p_pe = ffn_dim_p_pe,','.sampled=sampled,.epsilon=epsilon,')
pos=layout.index('    @export_name("X"')
layout=layout[:pos]+''' @export_name("X",[*]f16,true);@export_name("W",[*]f16,true);@export_name("Q_weight",[*]f16,true);
 @export_name("result",[*]f16,true);@export_name("normalized",[*]f16,true);@export_name("history",[*]f16,true);
 @export_name("progress",[*]u16,true);@export_name("timing",[*]u16,true);@export_name("queues",[*]u16,true);
 @export_name("init_task",fn()void);@export_name("hls_main",fn()void);
}
'''
p=runtime/'normalized_matmul_layout.csl';assert not p.exists();p.write_text(layout)
