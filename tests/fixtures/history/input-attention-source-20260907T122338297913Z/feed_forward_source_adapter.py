"""Adapt an unsealed source MLP probe into RMS(Z)->MLP->Z+MLP, before freezing."""

import difflib, json, shutil
from pathlib import Path
import numpy as np


def adapt(root, upstream, runtime, m, n, f, p, pack_tiles, *, blocked_upper=False):
    read = lambda name: json.loads((root / name).read_text())
    text = (root / "prefill.csl").read_text()
    a = text.index("fn rmsnorm_z() void {")
    b = text.index("fn z1_matmul()", a)
    text = text[:a] + """fn rmsnorm_z() void {
 rms_local.square_sum(ptr_Z,ptr_seqLen_dim_tmp,ptr_local_sum);
 comm_mod.mv_allreduce_add_x(ptr_local_sum);
 rms_local.inverse(ptr_local_sum);
 rms_local.normalize(ptr_Z,ptr_W,ptr_Z_norm,ptr_local_sum);
 prefill_struct();
}

""" + text[b:]
    needle = "fn prefill_struct() void {\n if(hls_phase==0)"
    assert text.count(needle) == 1
    text = text.replace(
        needle,
        """fn prefill_struct() void {
 if(hls_phase==-1){@fmovh(hls_normalized_dsd,@set_dsd_base_addr(Z_dsd,ptr_Z_norm));hls_phase=0;z1_matmul();}
 else if(hls_phase==0)""",
    )
    needle = "fn hls_main() void {hls_phase=0;timestamp.enable_tsc();timestamp.get_timestamp(&hls_start);z1_matmul();}"
    assert text.count(needle) == 1
    text = text.replace(
        needle,
        needle.replace("hls_phase=0", "hls_phase=-1").replace(
            "z1_matmul();", "rmsnorm_z();"
        ),
    )
    needle = "fn hls_finish() void {block_accum.snapshot();"
    assert text.count(needle) == 1
    text = text.replace(needle, needle + "add_result();")
    text += """
const rms_local=@import_module("rms_local.csl",.{.rows=seq_len_p_pe,.features=dim_p_pe,.global_features=dim_p_pe*P,.epsilon=eps});
var hls_normalized=@zeros([seq_len_p_pe*dim_p_pe]f16);
const hls_normalized_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*dim_p_pe}->hls_normalized[i]});
var hls_nptr:[*]f16=&hls_normalized;
comptime {@export_symbol(ptr_Z,"z");@export_symbol(ptr_W,"gamma");@export_symbol(ptr_Z,"result");@export_symbol(hls_nptr,"normalized");}
"""
    if blocked_upper:
        for name in ("z1", "z2"):
            needle = f"fn {name}_matmul() void {{"
            assert text.count(needle) == 1
            text = text.replace(needle, needle + "hidden_accum.reset();")
        needle = "    if (step < P) {\n        if(hls_phase==3){@fmovh(h2_dsd,0.0);}"
        assert text.count(needle) == 1
        text = text.replace(
            needle,
            needle
            + "\n if(hls_phase==0){@fmovh(z1_dsd,0.0);}else if(hls_phase==1){@fmovh(z2_dsd,0.0);}",
        )
        needle = "        if(HLS_SAMPLED){"
        assert text.count(needle) == 1
        text = text.replace(
            needle,
            "if(hls_phase==0){hidden_accum.merge(ptr_z1);}else if(hls_phase==1){hidden_accum.merge(ptr_z2);}"
            + needle,
        )
        for phase, name in ((0, "up"), (1, "gate")):
            needle = f"if(hls_phase=={phase}){{"
            # Only the phase controller owns persistent snapshot handoff.
            start = text.index("fn prefill_struct() void {")
            offset = text.index(needle, start) + len(needle)
            text = (
                text[:offset]
                + f"hidden_accum.snapshot();@mov32(hls_{name}_words_dsd,hidden_words_dsd);"
                + text[offset:]
            )
        text += '\nconst hidden_accum=@import_module("block_accumulate.csl",.{.length=seq_len_p_pe*ffn_dim_p_pe});\nconst hidden_words_dsd=@get_dsd(mem1d_dsd,.{.base_address=hidden_accum.words,.extent=@as(u16,seq_len_p_pe*ffn_dim_p_pe)});\n'
        for name in ("up", "gate"):
            text += f'var hls_{name}_words=@zeros([seq_len_p_pe*ffn_dim_p_pe]u32);\nconst hls_{name}_words_dsd=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{seq_len_p_pe*ffn_dim_p_pe}}->hls_{name}_words[i]}});\nvar hls_{name}_words_ptr:[*]u32=&hls_{name}_words;\ncomptime {{@export_symbol(hls_{name}_words_ptr,"{name}_accumulator");}}\n'
    (root / "prefill.csl").write_text(text)
    (root / "source-adapter.diff").write_text(
        "".join(
            difflib.unified_diff(
                (upstream / "prefill.csl").read_text().splitlines(True),
                text.splitlines(True),
                fromfile="pinned/prefill.csl",
                tofile="feed-forward/prefill.csl",
            )
        )
    )
    layout = (root / "layout.csl").read_text()
    i = layout.rfind("}")
    exports = "".join(
        f'@export_name("{name}",[*]f16,true);'
        for name in ("z", "gamma", "result", "normalized")
    )
    if blocked_upper:
        exports += "".join(
            f'@export_name("{name}_accumulator",[*]u32,true);'
            for name in ("up", "gate")
        )
    (root / "layout.csl").write_text(layout[:i] + exports + layout[i:])
    shutil.copyfile(runtime / "rms_local.csl", root / "rms_local.csl")
    shutil.copyfile(Path(__file__), root / "feed_forward_source_adapter.py")
    bs = read("inputs.json")[:3]
    logical = read("logical-inputs.json")[:3]
    rng = np.random.default_rng(210113)
    for b, l in zip(bs, logical):
        b["z"] = b.pop("x")
        l["z"] = l.pop("x")
        gamma = rng.uniform(0.5, 1.5, (1, n)).astype(np.float16).astype(float)
        l["gamma"] = gamma.ravel().tolist()
        b["gamma"] = np.broadcast_to(
            gamma.reshape(1, p, n // p), (p, p, n // p)
        ).tolist()
        # Bound up/gate inputs for the next compiler contract; exact half data.
        for key in ("up_weight", "gate_weight", "down_weight"):
            for obj in (b, l):
                obj[key] = (
                    (np.asarray(obj[key]) / 16)
                    .astype(np.float16)
                    .astype(float)
                    .tolist()
                )
    schema = read("schema.json")
    schema["inputs"]["z"] = schema["inputs"].pop("x")
    schema["inputs"]["gamma"] = n // p
    schema["outputs"].update(result=(m // p) * (n // p), normalized=(m // p) * (n // p))
    if blocked_upper:
        for name in ("up", "gate"):
            schema["outputs"][name + "_accumulator"] = (m // p) * (f // p)
            schema.setdefault("output_word_bits", {})[name + "_accumulator"] = 32
    for name, value in (
        ("inputs.json", bs),
        ("logical-inputs.json", logical),
        ("schema.json", schema),
    ):
        (root / name).write_text(json.dumps(value) + "\n")
