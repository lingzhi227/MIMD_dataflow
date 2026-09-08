"""One-time composed runtime authoring; shared math is a CSL library."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


from pathlib import Path

ROOT = repository_root(__file__)
rt = ROOT / "runtime/csl"
assert not (rt / "score_softmax_pe.csl").exists()
pe = (rt / "score_pe.csl").read_text()
pe = "param scale:f16;\n" + pe
needle = "fn hls_finish() void {progress[2]+=1;"
assert pe.count(needle) == 1
pe = pe.replace(needle, "fn hls_finish() void {softmax_main();progress[2]+=1;")
pe += """
const softmax=@import_module("softmax_local.csl",.{.rows=seq_len_p_pe,.cols=seq_len_p_pe});
var peaks=@zeros([seq_len_p_pe]f16);var sums=@zeros([seq_len_p_pe]f16);
var logits=@zeros([if(sampled!=0) S else 1]f16);var softmax_history=@zeros([if(sampled!=0) 5*seq_len_p_pe else 1]f16);var softmax_progress=@zeros([6]u16);
const logits_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->logits[i]});const score_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->score[i]});
const softmax_hd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe}->softmax_history[i]});
fn softmax_observe(phase:i16,values:[*]f16) void {if(sampled!=0){@fmovh(@increment_dsd_offset(softmax_hd,phase*seq_len_p_pe,f16),@set_dsd_base_addr(softmax_hd,values));}softmax_progress[phase]=1;}
fn softmax_main() void {
 for(@range(i16,6)) |i| {softmax_progress[i]=0;}
 if(sampled!=0){@fmovh(logits_dsd,score_dsd);}
 softmax.scale_max(ptr_score,&peaks,scale);softmax_observe(0,&peaks);
 comm_mod.mv_allreduce_max_x(&peaks);softmax_observe(1,&peaks);
 softmax.shift_exp(ptr_score,&peaks,ptr_seqLen_seqLen_tmp);
 softmax.local_sum(ptr_seqLen_seqLen_tmp,&sums);softmax_observe(2,&sums);
 comm_mod.mv_allreduce_add_x(&sums);softmax_observe(3,&sums);
 softmax.normalize(ptr_score,ptr_seqLen_seqLen_tmp,&sums);softmax_observe(4,&sums);softmax_progress[5]=1;
}
var logits_ptr:[*]f16=&logits;var exponents_ptr:[*]f16=if(sampled!=0) &seqLen_seqLen_tmp else &logits;var softmax_hp:[*]f16=&softmax_history;var softmax_pp:[*]u16=&softmax_progress;
comptime {@export_symbol(logits_ptr,"logits");@export_symbol(exponents_ptr,"exponents");@export_symbol(softmax_hp,"softmax_history");@export_symbol(softmax_pp,"softmax_progress");}
"""
(rt / "score_softmax_pe.csl").write_text(pe)
layout = (rt / "score_layout.csl").read_text()
layout = "param scale_bits:u16;\nconst scale=@bitcast(f16,scale_bits);\n" + layout
needle = ".sampled=sampled,"
assert layout.count(needle) == 2
layout = layout.replace(needle, needle + ".scale=scale,")
pos = layout.rfind("}")
layout = (
    layout[:pos]
    + """@export_name("logits",[*]f16,true);@export_name("exponents",[*]f16,true);@export_name("softmax_history",[*]f16,true);@export_name("softmax_progress",[*]u16,true);
"""
    + layout[pos:]
)
(rt / "score_softmax_layout.csl").write_text(layout)
