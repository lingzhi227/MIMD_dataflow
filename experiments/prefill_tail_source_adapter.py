"""Add original h1_matmul/z_add before an unsealed source FFN control."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import difflib, json, shutil, sys
from pathlib import Path
import numpy as np


def adapt(root, upstream, m, n, f, p):
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo), str(repo / "lib")]
    from prefill_tail_fixtures import batches
    from mesh_mlp_sdk import packed

    text = (root / "prefill.csl").read_text()
    needle = "fn prefill_struct() void {\n if(hls_phase==-1)"
    assert text.count(needle) == 1
    text = text.replace(
        needle,
        """fn prefill_struct() void {
 if(hls_phase==-3){@fmovh(hls_projection_dsd,h1_dsd);hls_phase=-2;z_add();}
 else if(hls_phase==-2){@fmovh(hls_z_snapshot_dsd,Z_dsd);hls_phase=-1;rmsnorm_z();}
 else if(hls_phase==-1)""",
    )
    needle = "fn hls_main() void {hls_phase=-1;timestamp.enable_tsc();timestamp.get_timestamp(&hls_start);rmsnorm_z();}"
    assert text.count(needle) == 1
    text = text.replace(
        needle,
        needle.replace("hls_phase=-1", "hls_phase=-3").replace(
            "rmsnorm_z();", "h1_matmul();"
        ),
    )
    text += """
var hls_projection=@zeros([seq_len_p_pe*dim_p_pe]f16);
var hls_z_snapshot=@zeros([seq_len_p_pe*dim_p_pe]f16);
const hls_projection_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*dim_p_pe}->hls_projection[i]});
const hls_z_snapshot_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*dim_p_pe}->hls_z_snapshot[i]});
var hls_projection_ptr:[*]f16=&hls_projection;var hls_z_snapshot_ptr:[*]f16=&hls_z_snapshot;
comptime {@export_symbol(ptr_output,"attention");@export_symbol(ptr_O_weight,"output_weight");@export_symbol(ptr_X,"residual");@export_symbol(hls_projection_ptr,"projection");@export_symbol(hls_z_snapshot_ptr,"post_projection_residual");}
"""
    (root / "prefill.csl").write_text(text)
    (root / "source-adapter.diff").write_text(
        "".join(
            difflib.unified_diff(
                (upstream / "prefill.csl").read_text().splitlines(True),
                text.splitlines(True),
                fromfile="pinned/prefill.csl",
                tofile="supplied-attention-tail/prefill.csl",
            )
        )
    )
    layout = (root / "layout.csl").read_text()
    i = layout.rfind("}")
    exports = "".join(
        f'@export_name("{name}",[*]f16,true);'
        for name in (
            "attention",
            "output_weight",
            "residual",
            "projection",
            "post_projection_residual",
        )
    )
    (root / "layout.csl").write_text(layout[:i] + exports + layout[i:])
    logical = batches(m, n, f, p)[:3]
    physical = []
    for b in logical:
        a = np.asarray(b["attention"]).reshape(m, n)
        o = np.asarray(b["output_weight"]).reshape(n, n)
        r = np.asarray(b["residual"]).reshape(m, n)
        u = np.asarray(b["up_weight"]).reshape(n, f)
        g = np.asarray(b["gate_weight"]).reshape(n, f)
        d = np.asarray(b["down_weight"]).reshape(f, n)
        row = {k: v.tolist() for k, v in packed(dict(P=p), (a, u, g, d)).items()}
        row["attention"] = row.pop("x")
        row["output_weight"] = packed(dict(P=p), (a, o, o, o))["up_weight"].tolist()
        from mesh_common import pack_tiles

        row["residual"] = pack_tiles(r, p, p, "F").tolist()
        row["gamma"] = np.broadcast_to(
            np.asarray(b["gamma"]).reshape(1, p, n // p), (p, p, n // p)
        ).tolist()
        physical.append(row)
    schema = json.loads((root / "schema.json").read_text())
    del schema["inputs"]["z"]
    schema["inputs"].update(
        attention=(m // p) * (n // p),
        output_weight=(n // p) ** 2,
        residual=(m // p) * (n // p),
    )
    schema["outputs"].update(
        projection=(m // p) * (n // p), post_projection_residual=(m // p) * (n // p)
    )
    for name, value in (
        ("inputs.json", physical),
        ("logical-inputs.json", logical),
        ("schema.json", schema),
    ):
        (root / name).write_text(json.dumps(value) + "\n")
    shutil.copyfile(Path(__file__), root / "prefill_tail_source_adapter.py")
    shutil.copyfile(
        repo / "tests/support/prefill_tail_fixtures.py", root / "tests/support/prefill_tail_fixtures.py"
    )
    for name in ("tests/support/feed_forward_fixtures.py", "tests/support/mlp_fixtures.py"):
        shutil.copyfile(repo / name, root / name)
