"""One-time authoring of resident composition; not part of compiler/codegen."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

from pathlib import Path
import re
R=Path(__file__).resolve().parents[1]/'runtime/csl'
assert not (R/'attention_pe.csl').exists()
s=(R/'score_softmax_pe.csl').read_text();v=(R/'device_matmul_pe.csl').read_text()
s=s.replace('task left_matrix_finish() void {@block(left_matrix_finish_id);@unblock(two_hop_comm_finish_id);}', 'task left_matrix_finish() void {@block(left_matrix_finish_id);if(in_preshift){left_matrix_shift_callback();}else{@unblock(two_hop_comm_finish_id);}}')
s=s.replace('task right_matrix_finish() void {@block(right_matrix_finish_id);@activate(two_hop_comm_finish_id);}', 'task right_matrix_finish() void {@block(right_matrix_finish_id);if(value_phase){value_shift();return;}@activate(two_hop_comm_finish_id);}')
s=s.replace('task next_step() void {@block(next_step_id);matmul_T_compute();}', 'task next_step() void {@block(next_step_id);if(is_T_matmul){matmul_T_compute();}else{matmul_compute();}}')
s=s.replace('fn hls_main() void {timestamp.enable_tsc();', 'fn hls_main() void {for(@range(i16,3)) |i| {value_progress[i]=0;}right_matrix_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->dummy[i]});@fmovh(bw,bd);timestamp.enable_tsc();')
# Include private V copy inside the measured interval, as with the K copy.
s=s.replace('@fmovh(bw,bd);timestamp.enable_tsc();timestamp.get_timestamp(&start);', 'timestamp.enable_tsc();timestamp.get_timestamp(&start);@fmovh(bw,bd);')
s=s.replace('fn hls_finish() void {softmax_main();progress[2]+=1;', 'fn hls_finish() void {if(is_T_matmul){softmax_main();if(sampled!=0){@fmovh(probability_dsd,score_dsd);@fmovh(exponent_copy_dsd,seqLen_seqLen_tmp_dsd);}output_matmul();return;}progress[2]+=1;value_progress[3]+=1;')
s=s.replace('@export_symbol(ptr_score,"result");','@export_symbol(result_ptr,"result");')
s=s.replace('var exponents_ptr:[*]f16=if(sampled!=0) &seqLen_seqLen_tmp else &logits;', 'var exponents_ptr:[*]f16=&exponent_copy;')
start=v.index('fn left_matrix_shift_callback()');end=v.index('fn matmul_map_func',start)
body=v[start:end]+v[v.index('fn matmul_compute()'):v.index('fn hls_main()')]
for old,new in [('progress','value_progress'),('history','value_history'),('left_owners','value_left'),('right_owners','value_right'),('hd','value_hd'),('ld','value_ld'),('rd','value_rd')]:
 body=re.sub(r'\b'+old+r'\b',new,body)
s+='''
// Phase-scoped reuse: seqLen_dim_tmp is dead K receive storage; seqLen_seqLen_tmp
// is dead softmax exponent storage. Observations are copied before destructive reuse.
var ptr_left_matrix_recv:[*]f16;
var left_dim_length:i16=0;var in_preshift:bool=false;var pre_remaining:i16=0;var shift_round:i16=0;
var value_phase:bool=false;var value_remaining:i16=0;var value_step:i16=0;
var V_input=@zeros([L]f16);var XV_tile=@zeros([L]f16);var output_tile=@zeros([L]f16);
const bd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->V_input[i]});const bw=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->XV_tile[i]});
const aw=score_dsd;const output_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->output_tile[i]});
var value_history=@zeros([if(sampled!=0) P*L else 1]f16);var value_left=@zeros([if(sampled!=0) P*S else 1]f16);var value_right=@zeros([if(sampled!=0) P*L else 1]f16);
const value_hd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->value_history[i]});const value_ld=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->value_left[i]});const value_rd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{L}->value_right[i]});
var probability=@zeros([if(sampled!=0) S else 1]f16);var exponent_copy=@zeros([if(sampled!=0) S else 1]f16);var value_progress=@zeros([4]u16);
const probability_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->probability[i]});const exponent_copy_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->exponent_copy[i]});
var v_ptr:[*]f16=&V_input;var result_ptr:[*]f16=&output_tile;var prob_ptr:[*]f16=&probability;var vh_ptr:[*]f16=&value_history;var vl_ptr:[*]f16=&value_left;var vr_ptr:[*]f16=&value_right;var vp_ptr:[*]u16=&value_progress;
comptime {@export_symbol(v_ptr,"v");@export_symbol(prob_ptr,"probability");@export_symbol(vh_ptr,"value_history");@export_symbol(vl_ptr,"value_left");@export_symbol(vr_ptr,"value_right");@export_symbol(vp_ptr,"value_progress");}
'''+body
(R/'attention_pe.csl').write_text(s)
s=(R/'score_softmax_layout.csl').read_text();pos=s.rfind('}')
s=s[:pos]+'''@export_name("v",[*]f16,true);@export_name("probability",[*]f16,true);@export_name("value_history",[*]f16,true);@export_name("value_left",[*]f16,true);@export_name("value_right",[*]f16,true);@export_name("value_progress",[*]u16,true);
'''+s[pos:]
(R/'attention_layout.csl').write_text(s)
