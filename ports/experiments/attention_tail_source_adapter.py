"""Join original score/softmax/device-aligned value before an unsealed source tail."""

import difflib, json, shutil, sys
from pathlib import Path
import numpy as np
from attention_layout_adapter import align_column_major_value
from resident_attention_source_probe import repair_softmax_maximum


def adapt(root, upstream, m, n, f, p):
    repo = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(repo), str(repo / "toolchain")]
    from attention_tail_fixtures import batches
    from mesh_mlp_sdk import packed
    from mesh_common import pack_tiles

    text = (root / "prefill.csl").read_text()
    marker = "fn prefill_struct() void {\n if(hls_phase==-3)"
    assert text.count(marker) == 1
    text = text.replace(
        marker,
        """fn prefill_struct() void {
 if(hls_phase==-6){@fmovh(hls_logits_dsd,score_dsd);hls_phase=-5;softmax_score();}
 else if(hls_phase==-5){@fmovh(hls_probability_dsd,score_dsd);hls_phase=-4;output_matmul();}
 else if(hls_phase==-4){@fmovh(hls_attention_dsd,output_dsd);hls_phase=-3;h1_matmul();}
 else if(hls_phase==-3)""",
    )
    marker = "fn hls_main() void {hls_phase=-3;timestamp.enable_tsc();timestamp.get_timestamp(&hls_start);h1_matmul();}"
    assert text.count(marker) == 1
    text = text.replace(
        marker,
        """fn hls_main() void {hls_phase=-6;hls_scale[0]=alpha;right_matrix_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->dummy[i]});timestamp.enable_tsc();timestamp.get_timestamp(&hls_start);score_matmul();}""",
    )
    first = text.index("fn softmax_score()")
    last = text.index("fn output_matmul()", first)
    text = text[:first] + repair_softmax_maximum(text[first:last]) + text[last:]
    text = align_column_major_value(text, value_phase_condition="hls_phase==-4")
    marker = "fn h1_matmul() void {"
    assert text.count(marker) == 1
    text = text.replace(
        marker,
        marker
        + "\n right_matrix_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->dummy[i]});",
    )
    marker = '@export_symbol(ptr_output,"attention");'
    assert text.count(marker) == 1
    text = text.replace(marker, "")
    text += """
var hls_logits=@zeros([seq_len_p_pe*seq_len_p_pe]f16);var hls_probability=@zeros([seq_len_p_pe*seq_len_p_pe]f16);var hls_attention=@zeros([seq_len_p_pe*dim_p_pe]f16);var hls_scale=@zeros([1]f16);
const hls_logits_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*seq_len_p_pe}->hls_logits[i]});
const hls_probability_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*seq_len_p_pe}->hls_probability[i]});
const hls_attention_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*dim_p_pe}->hls_attention[i]});
var hls_logits_ptr:[*]f16=&hls_logits;var hls_probability_ptr:[*]f16=&hls_probability;var hls_attention_ptr:[*]f16=&hls_attention;var hls_scale_ptr:[*]f16=&hls_scale;
comptime {@export_symbol(ptr_XQ,"q");@export_symbol(ptr_XK,"k");@export_symbol(ptr_XV,"v");@export_symbol(hls_logits_ptr,"logits");@export_symbol(hls_probability_ptr,"probability");@export_symbol(hls_attention_ptr,"attention");@export_symbol(hls_scale_ptr,"scale");}
"""
    (root / "prefill.csl").write_text(text)
    (root / "source-adapter.diff").write_text(
        "".join(
            difflib.unified_diff(
                (upstream / "prefill.csl").read_text().splitlines(True),
                text.splitlines(True),
                fromfile="pinned/prefill.csl",
                tofile="resident-qkv-attention-tail/prefill.csl",
            )
        )
    )
    text = (root / "layout.csl").read_text()
    pos = text.rfind("}")
    extra = "".join(
        f'@export_name("{key}",[*]f16,true);'
        for key in ("q", "k", "v", "logits", "probability", "scale")
    )
    (root / "layout.csl").write_text(text[:pos] + extra + text[pos:])
    logical = batches(m, n, f, p)[:3]
    physical = []
    for b in logical:
        u = np.asarray(b["up_weight"]).reshape(n, f)
        g = np.asarray(b["gate_weight"]).reshape(n, f)
        d = np.asarray(b["down_weight"]).reshape(f, n)
        o = np.asarray(b["output_weight"]).reshape(n, n)
        dummy = np.zeros((m, n))
        row = {
            key: val.tolist()
            for key, val in packed(dict(P=p), (dummy, u, g, d)).items()
            if key != "x"
        }
        row["output_weight"] = packed(dict(P=p), (dummy, o, o, o))["up_weight"].tolist()
        for key in ("q", "k", "v", "residual"):
            row[key] = pack_tiles(np.asarray(b[key]).reshape(m, n), p, p, "F").tolist()
        row["gamma"] = np.broadcast_to(
            np.asarray(b["gamma"]).reshape(1, p, n // p), (p, p, n // p)
        ).tolist()
        physical.append(row)
    schema = json.loads((root / "schema.json").read_text())
    del schema["inputs"]["attention"]
    schema["inputs"].update({key: (m // p) * (n // p) for key in ("q", "k", "v")})
    schema["outputs"].update(
        logits=(m // p) ** 2,
        probability=(m // p) ** 2,
        attention=(m // p) * (n // p),
        scale=1,
    )
    for name, obj in [
        ("logical-inputs.json", logical),
        ("inputs.json", physical),
        ("schema.json", schema),
    ]:
        (root / name).write_text(json.dumps(obj) + "\n")
    for name in (
        "attention_tail_fixtures.py",
        "prefill_tail_fixtures.py",
        "feed_forward_fixtures.py",
        "mlp_fixtures.py",
    ):
        if not (root / name).exists():
            shutil.copyfile(repo / name, root / name)
    for name in (
        "attention_tail_source_adapter.py",
        "attention_layout_adapter.py",
        "resident_attention_source_probe.py",
    ):
        shutil.copyfile(repo / "experiments" / name, root / name)
