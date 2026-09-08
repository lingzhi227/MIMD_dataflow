"""One-time runtime authoring from reviewed resident CSL; never called by codegen."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import hashlib, json
from pathlib import Path

ROOT = repository_root(__file__)
rt = ROOT / "runtime/csl"
pe = (rt / "normalized_matmul_pe.csl").read_text()
layout = (rt / "normalized_matmul_layout.csl").read_text()
pe = pe.replace("param sampled:i16;", "param sampled:i16;param projections:i16;")
pe = pe.replace(
    "var Q_weight_tile: [dim_p_pe * dim_p_pe]f16 = @zeros([dim_p_pe * dim_p_pe]f16);",
    "var Q_weight_tile: [projections*dim_p_pe * dim_p_pe]f16 = @zeros([projections*dim_p_pe * dim_p_pe]f16);",
)
pe = pe.replace(
    "var XQ_tile: [seq_len_p_pe * dim_p_pe]f16 = @zeros([seq_len_p_pe * dim_p_pe]f16);",
    "var XQ_tile: [projections*seq_len_p_pe * dim_p_pe]f16 = @zeros([projections*seq_len_p_pe * dim_p_pe]f16);",
)
needle = "        if(sampled!=0){const hd=@increment_dsd_offset(hls_hist_dsd,step*seq_len_p_pe*dim_p_pe,f16);@fmovh(hd,XQ_dsd);}"
assert pe.count(needle) == 1
pe = pe.replace(
    needle,
    "        if(sampled!=0){const hd=@increment_dsd_offset(hls_hist_dsd,(hls_branch*P+step)*seq_len_p_pe*dim_p_pe,f16);const value=@set_dsd_base_addr(XQ_dsd,ptr_out_matrix);@fmovh(hd,value);}",
)
a = pe.index("fn xq_matmul()")
b = pe.index("\n\nconst config=", a)
body = pe[a:b]
body = body.replace(
    "    // Clearing\n    @load_to_dsr(comp_dest_dsr_1, XQ_dsd);",
    "    ptr_out_matrix = &XQ_tile[hls_branch*seq_len_p_pe*dim_p_pe];\n    const clear=@set_dsd_base_addr(XQ_dsd,ptr_out_matrix);\n    @load_to_dsr(comp_dest_dsr_1, clear);",
)
body = (
    body.replace(
        "    ptr_left_matrix_send = &seqLen_dim_tmp;\n    ptr_left_matrix_recv = &X_norm_tile;",
        """    if(hls_branch==0){
      ptr_left_matrix_send = &seqLen_dim_tmp;ptr_left_matrix_recv = &X_norm_tile;
    }else{
      swap_ptr=ptr_left_matrix_send;ptr_left_matrix_send=ptr_left_matrix_recv;ptr_left_matrix_recv=swap_ptr;
    }""",
    )
    .replace(
        "    ptr_right_matrix_recv = &Q_weight_tile;",
        "    ptr_right_matrix_recv = &Q_weight_tile[hls_branch*dim_p_pe*dim_p_pe];",
    )
    .replace("    ptr_out_matrix = &XQ_tile;", "")
)
body = body.replace(
    "    in_preshift = true;\n    pre_remaining = offset_step;\n    shift_round = 0;\n    left_matrix_shift_callback();",
    """    if(hls_branch==0){
      in_preshift=true;pre_remaining=offset_step;shift_round=0;left_matrix_shift_callback();
    }else{matmul_compute();}""",
)
pe = pe[:a] + body + pe[b:]
pe = pe.replace(
    "var hls_history=@zeros([if(sampled!=0) P*seq_len_p_pe*dim_p_pe else 1]f16);",
    "var hls_history=@zeros([if(sampled!=0) projections*P*seq_len_p_pe*dim_p_pe else 1]f16);\nvar hls_reuse=@zeros([if(sampled!=0) projections*seq_len_p_pe*dim_p_pe else 1]f16);\nconst hls_reuse_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*dim_p_pe}->hls_reuse[i]});\nvar hls_branch:i16=0;",
)
pe = pe.replace("fn hls_main() void {", "fn hls_main() void {\n hls_branch=0;")
pe = pe.replace(
    " hls_progress[2]=1;hls_progress[3]+=1;",
    """ if(sampled!=0){
   const live=@set_dsd_base_addr(normalized_dsd,ptr_left_matrix_send);
   @fmovh(@increment_dsd_offset(hls_reuse_dsd,hls_branch*seq_len_p_pe*dim_p_pe,f16),live);
 }
 hls_branch+=1;hls_progress[2]=@as(u16,hls_branch);
 if(hls_branch<projections){xq_matmul();return;}
 hls_progress[3]+=1;""",
)
pe = pe.replace(
    "var hp:[*]f16=&hls_history;",
    "var reuse_ptr:[*]f16=&hls_reuse;\nvar hp:[*]f16=&hls_history;",
).replace(
    ' @export_symbol(ptr_X,"X");',
    ' @export_symbol(reuse_ptr,"reuse");\n @export_symbol(ptr_X,"X");',
)
layout = layout.replace(
    "param sampled:i16;", "param sampled:i16;param projections:i16;"
)
assert ".sampled = sampled" in layout or ".sampled=sampled" in layout
layout = layout.replace(
    ".sampled=sampled", ".sampled=sampled,.projections=projections"
).replace(".sampled = sampled", ".sampled = sampled, .projections=projections")
pos = layout.rfind("}")
layout = layout[:pos] + ' @export_name("reuse",[*]f16,true);\n' + layout[pos:]
pe = pe.replace(
    "&XQ_tile[hls_branch*seq_len_p_pe*dim_p_pe]",
    "@ptrcast([*]f16,&XQ_tile[hls_branch*seq_len_p_pe*dim_p_pe])",
).replace(
    "&Q_weight_tile[hls_branch*dim_p_pe*dim_p_pe]",
    "@ptrcast([*]f16,&Q_weight_tile[hls_branch*dim_p_pe*dim_p_pe])",
)
for name, content in [
    ("normalized_fanout_pe.csl", pe),
    ("normalized_fanout_layout.csl", layout),
]:
    path = rt / name
    assert (
        not path.exists()
    ), "one-time authoring: review existing files instead of overwrite"
    path.write_text(content)
print("Authored normalized fan-out runtime; not compiler-dispatched yet")
