"""Extract the actual production reverse scan into an SDK boundary probe."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


from pathlib import Path
import datetime, hashlib, json, shutil

ROOT = repository_root(__file__)
source = ROOT / "runtime/csl/spmv_pe.csl"
s = source.read_text()
a = s.index("fn compute_north_fn(")
b = s.index("// perform local compute using buffer for input x-vec (south)", a)
out = (
    ROOT
    / "validation/evidence"
    / (
        "reverse-scan-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
)
out.mkdir()
(out / "layout.csl").write_text(
    """const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});
layout {@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});@export_name("main",fn()void);@export_name("result",[*]f32,true);}
"""
)
(out / "pe.csl").write_text(
    """param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const max_local_nnz_rows:u16=1;
const local_vec_sz:u16=4;
var cols=[2]u16{0,3};var lens=[2]u16{1,1};var locs=[2]u16{0,1};var rows=[2]u16{0,0};var vals=[2]f32{2.0,5.0};var nnz=[1]u16{2};var nnzc=[1]u16{0};
const mat_col_idx_buf=&cols;const mat_col_len_buf=&lens;const mat_col_loc_buf=&locs;const mat_rows_buf=&rows;const mat_vals_buf=&vals;const local_nnz=&nnz;const local_nnz_cols=&nnzc;
var north_train_start_idx:u16=0;
var x=[4]f32{3.0,7.0,11.0,13.0};var partial=[1]f32{0.0};var yrows=[1]u16{0};var result=@zeros([8]f32);var p:[*]f32=&result;
"""
    + s[a:b]
    + """
fn main() void {
 nnzc[0]=0;north_train_start_idx=nnzc[0]-1;partial[0]=0.0;
 compute_north_fn(&partial,&yrows,&x,0,4);result[0]=partial[0];result[1]=@as(f32,north_train_start_idx);
 nnzc[0]=1;north_train_start_idx=nnzc[0]-1;
 compute_north_fn(&partial,&yrows,&x,0,4);result[2]=partial[0];result[3]=@as(f32,north_train_start_idx);
 nnzc[0]=2;north_train_start_idx=nnzc[0]-1;partial[0]=0.0;
 // For the upper segment x is indexed relative to its low bound.
 x[0]=11.0;x[1]=13.0;
 compute_north_fn(&partial,&yrows,&x,2,4);result[4]=partial[0];result[5]=@as(f32,north_train_start_idx);
 x[0]=3.0;x[1]=7.0;
 compute_north_fn(&partial,&yrows,&x,0,2);result[6]=partial[0];result[7]=@as(f32,north_train_start_idx);
 sys.unblock_cmd_stream();
}
comptime {@export_symbol(main);@export_symbol(p,"result");}
"""
)
shutil.copy2(Path(__file__).parent / "run.py", out / "run.py")
(out / "manifest.json").write_text(
    json.dumps(
        dict(
            source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            function_sha256=hashlib.sha256(s[a:b].encode()).hexdigest(),
            files={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.iterdir()
            },
        ),
        indent=2,
    )
)
print(out)
