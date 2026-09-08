"""Join source input RMS/QKV/pair stages before an unsealed attention-tail control."""

import difflib, json, shutil, sys
from pathlib import Path
import numpy as np


def adapt(root, upstream, m, n, f, p):
    assert n % (2 * p) == 0, "complete adjacent pairs on each PE"
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo), str(repo / "toolchain")]
    from input_attention_fixtures import batches
    from mesh_mlp_sdk import packed
    from mesh_common import pack_tiles

    text = (root / "prefill.csl").read_text()
    a = text.index("fn rmsnorm_x() void {")
    b = text.index("fn xq_matmul()", a)
    text = text[:a] + """fn rmsnorm_x() void {
 rms_local.square_sum(ptr_X,ptr_seqLen_dim_tmp,ptr_local_sum);
 comm_mod.mv_allreduce_add_x(ptr_local_sum);rms_local.inverse(ptr_local_sum);
 rms_local.normalize(ptr_X,ptr_W,ptr_X_norm,ptr_local_sum);prefill_struct();
}

""" + text[b:]
    for name, next_name in (("xk_matmul", "xv_matmul"), ("xv_matmul", "xq_rope")):
        a = text.index("fn " + name + "()")
        b = text.index("fn " + next_name + "()", a)
        body = text[a:b]
        marker = "    ptr_left_matrix_send = &seqLen_dim_tmp;\n    ptr_left_matrix_recv = &X_norm_tile;"
        assert body.count(marker) == 1
        body = body.replace(
            marker,
            "    swap_ptr=ptr_left_matrix_send;\n    ptr_left_matrix_send=ptr_left_matrix_recv;\n    ptr_left_matrix_recv=swap_ptr;",
        )
        text = text[:a] + body + text[b:]
    for i in range(1, 5):
        marker = f"var X_tmp_{i}: [_dim_p_pe / 2]f16 = @zeros([_dim_p_pe / 2]f16);"
        assert text.count(marker) == 1
        text = text.replace(marker, marker.replace("_dim_p_pe / 2", "seq_len_p_pe"))
        marker = f"|i|{{_dim_p_pe / 2}} -> X_tmp_{i}[i]"
        assert text.count(marker) == 1
        text = text.replace(marker, marker.replace("_dim_p_pe / 2", "seq_len_p_pe"))
    marker = "fn prefill_struct() void {\n if(hls_phase==-6)"
    assert text.count(marker) == 1
    text = text.replace(
        marker,
        """fn prefill_struct() void {
 if(hls_phase==-12){@fmovh(hls_input_normalized_dsd,@set_dsd_base_addr(X_dsd,ptr_X_norm));hls_phase=-11;xq_matmul();}
 else if(hls_phase==-11){@fmovh(hls_q_raw_dsd,XQ_dsd);hls_phase=-10;xk_matmul();}
 else if(hls_phase==-10){@fmovh(hls_k_raw_dsd,XK_dsd);hls_phase=-9;xv_matmul();}
 else if(hls_phase==-9){@fmovh(hls_v_raw_dsd,XV_dsd);hls_phase=-8;xq_rope();}
 else if(hls_phase==-8){@fmovh(hls_q_rotated_dsd,XQ_dsd);hls_phase=-7;xk_rope();}
 else if(hls_phase==-7){@fmovh(hls_k_rotated_dsd,XK_dsd);hls_phase=-6;score_matmul();}
 else if(hls_phase==-6)""",
    )
    marker = "fn hls_main() void {hls_phase=-6;hls_scale[0]=alpha;right_matrix_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->dummy[i]});timestamp.enable_tsc();timestamp.get_timestamp(&hls_start);score_matmul();}"
    assert text.count(marker) == 1
    text = text.replace(
        marker,
        marker.replace("hls_phase=-6", "hls_phase=-12").replace(
            "score_matmul();", "rmsnorm_x();"
        ),
    )
    for name in ("q", "k", "v"):
        marker = f'@export_symbol(ptr_X{name.upper()},"{name}");'
        assert text.count(marker) == 1
        text = text.replace(marker, "")
    marker = '@export_symbol(ptr_X,"residual");'
    assert text.count(marker) == 1
    text = text.replace(marker, '@export_symbol(ptr_X,"input_x");')
    snapshots = (
        "input_normalized",
        "q_raw",
        "k_raw",
        "v_raw",
        "q_rotated",
        "k_rotated",
    )
    for name in snapshots:
        text += f"""\nvar hls_{name}=@zeros([seq_len_p_pe*dim_p_pe]f16);
const hls_{name}_dsd=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{seq_len_p_pe*dim_p_pe}}->hls_{name}[i]}});
var hls_{name}_ptr:[*]f16=&hls_{name};
comptime {{@export_symbol(hls_{name}_ptr,"{name}");}}
"""
    for name, pointer in (
        ("q_weight", "ptr_Q_weight"),
        ("k_weight", "ptr_K_weight"),
        ("v_weight", "ptr_V_weight"),
        ("cosine", "ptr_freqs_cos"),
        ("sine", "ptr_freqs_sin"),
    ):
        text += f'\ncomptime {{@export_symbol({pointer},"{name}");}}\n'
    (root / "prefill.csl").write_text(text)
    (root / "source-adapter.diff").write_text(
        "".join(
            difflib.unified_diff(
                (upstream / "prefill.csl").read_text().splitlines(True),
                text.splitlines(True),
                fromfile="pinned/prefill.csl",
                tofile="shared-gamma-input-attention/prefill.csl",
            )
        )
    )
    layout = (root / "layout.csl").read_text()
    for name in ("q", "k", "v", "residual"):
        marker = f'@export_name("{name}",[*]f16,true);'
        assert layout.count(marker) == 1
        layout = layout.replace(marker, "")
    pos = layout.rfind("}")
    exports = snapshots + (
        "input_x",
        "q_weight",
        "k_weight",
        "v_weight",
        "cosine",
        "sine",
    )
    layout = (
        layout[:pos]
        + "".join(f'@export_name("{name}",[*]f16,true);' for name in exports)
        + layout[pos:]
    )
    (root / "layout.csl").write_text(layout)
    logical = batches(m, n, f, p)[:3]
    physical = []
    dummy = np.zeros((m, n))
    mt, nt = m // p, n // p
    for b in logical:
        u = np.asarray(b["up_weight"]).reshape(n, f)
        g = np.asarray(b["gate_weight"]).reshape(n, f)
        d = np.asarray(b["down_weight"]).reshape(f, n)
        row = {
            key: value.tolist()
            for key, value in packed(dict(P=p), (dummy, u, g, d)).items()
            if key != "x"
        }
        for key in ("q_weight", "k_weight", "v_weight", "output_weight"):
            w = np.asarray(b[key]).reshape(n, n)
            row[key] = packed(dict(P=p), (dummy, w, w, w))["up_weight"].tolist()
        row["input_x"] = pack_tiles(
            np.asarray(b["input_x"]).reshape(m, n), p, p, "F"
        ).tolist()
        for key, length in (("gamma", nt), ("cosine", nt // 2), ("sine", nt // 2)):
            row[key] = np.broadcast_to(
                np.asarray(b[key]).reshape(1, p, length), (p, p, length)
            ).tolist()
        physical.append(row)
    schema = json.loads((root / "schema.json").read_text())
    for key in ("q", "k", "v", "residual"):
        del schema["inputs"][key]
    schema["inputs"].update(
        input_x=mt * nt,
        q_weight=nt * nt,
        k_weight=nt * nt,
        v_weight=nt * nt,
        cosine=nt // 2,
        sine=nt // 2,
    )
    schema["outputs"].update({name: mt * nt for name in snapshots})
    for name, value in (
        ("logical-inputs.json", logical),
        ("inputs.json", physical),
        ("schema.json", schema),
    ):
        (root / name).write_text(json.dumps(value) + "\n")
    for name in (
        "input_attention_fixtures.py",
        "attention_tail_fixtures.py",
        "prefill_tail_fixtures.py",
        "feed_forward_fixtures.py",
    ):
        if not (root / name).exists():
            shutil.copyfile(repo / name, root / name)
    shutil.copyfile(__file__, root / "input_attention_source_adapter.py")
