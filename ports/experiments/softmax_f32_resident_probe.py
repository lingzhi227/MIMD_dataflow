"""SDK primitive probe for resident f32 softmax/row communication.

Explicit unsealed CSL adapter over generated31-node diagnostics; not a typed
mixed HLS policy, nor a fix for the remaining half V/PV/O/Z path.
"""

import json, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain"), str(ROOT / "experiments")]
from input_attention_codegen_probe import prepare


def adapt(root, s):
    assert not (root / "provenance.json").exists()
    comm = (root / "inference_comm.csl").read_text()
    for name, end in (
        ("mv_allreduce_max_x", "fn mv_allreduce_add_x"),
        ("mv_allreduce_add_x", "// ================= matmul_T"),
    ):
        a = comm.index("fn " + name + "(")
        b = comm.index(end, a + 3)
        body = comm[a:b].replace(name, name + "_f32").replace("[*]f16", "[*]f32")
        body = body.replace(
            "    vector_buf_dsd = @set_dsd_base_addr(vector_buf_dsd, vector_buf_ptr);",
            "    const wide_vector_dsd=@get_dsd(mem1d_dsd,.{.base_address=vector_buf_ptr,.extent=@as(u16,seq_len_p_pe)});",
        )
        body = (
            body.replace("vector_buf_dsd", "wide_vector_dsd")
            .replace("@fmaxh", "@fmaxs")
            .replace("@fmovh", "@fmovs")
            .replace("@faddh", "@fadds")
            .replace("@mov16", "@mov32")
        )
        comm += (
            "\n// Synchronous f32 row extension; same source routing and DSR2 lease.\n"
            + body
        )
    (root / "inference_comm_wide_rows.csl").write_text(comm)
    shutil.copyfile(
        ROOT / "toolchain/runtime/softmax_f32_local.csl", root / "softmax_f32_local.csl"
    )
    pe = (
        (root / "pe.csl")
        .read_text()
        .replace('"inference_comm.csl"', '"inference_comm_wide_rows.csl"')
    )
    a = pe.index("fn attention_softmax() void {")
    b = pe.index("fn attention_value_setup()", a)
    pe = pe[:a] + """fn attention_softmax() void {
 wide_ops.convert(f32,f16,wide_values_view,attention_score_view,.{});
 wide_softmax.scale_max(&wide_probability,&wide_peaks,@as(f32,attention_scale));
 @fmovs(wide_history_view,wide_peaks_view);
 comm_mod.mv_allreduce_max_x_f32(&wide_peaks);
 @fmovs(@increment_dsd_offset(wide_history_view,seq_len_p_pe,f32),wide_peaks_view);
 wide_softmax.shift_exp(&wide_probability,&wide_peaks,&wide_exponents);
 wide_softmax.local_sum(&wide_exponents,&wide_sums);
 @fmovs(@increment_dsd_offset(wide_history_view,2*seq_len_p_pe,f32),wide_sums_view);
 comm_mod.mv_allreduce_add_x_f32(&wide_sums);
 @fmovs(@increment_dsd_offset(wide_history_view,3*seq_len_p_pe,f32),wide_sums_view);
 wide_softmax.normalize(&wide_probability,&wide_exponents,&wide_sums);
 @fmovs(@increment_dsd_offset(wide_history_view,4*seq_len_p_pe,f32),wide_sums_view);
 wide_ops.convert(f16,f32,attention_score_view,wide_values_view,.{});
 for(@range(i16,6)) |i| {attention_softmax_progress[i]=1;}
}
""" + pe[b:]
    pe += """
const wide_ops=@import_module("<dsd_ops>");
const wide_softmax=@import_module("softmax_f32_local.csl",.{.rows=seq_len_p_pe,.cols=seq_len_p_pe});
var wide_probability=@zeros([S]f32);var wide_exponents=@zeros([S]f32);
var wide_peaks=@zeros([seq_len_p_pe]f32);var wide_sums=@zeros([seq_len_p_pe]f32);
var wide_history=@zeros([5*seq_len_p_pe]f32);
const wide_values_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{S}->wide_probability[i]});
const wide_peaks_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe}->wide_peaks[i]});
const wide_sums_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe}->wide_sums[i]});
const wide_history_view=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe}->wide_history[i]});
"""
    lengths = dict(
        wide_probability=s["score_length"],
        wide_exponents=s["score_length"],
        wide_peaks=s["Mt"],
        wide_sums=s["Mt"],
        wide_history=5 * s["Mt"],
    )
    exports = ""
    for name in lengths:
        pe += f'\nvar {name}_ptr:[*]f32=&{name};\ncomptime {{@export_symbol({name}_ptr,"{name}");}}\n'
        exports += f'@export_name("{name}",[*]f32,true);\n'
    (root / "pe.csl").write_text(pe)
    layout = (root / "layout.csl").read_text()
    a = layout.rfind("}")
    (root / "layout.csl").write_text(layout[:a] + exports + layout[a:])
    schema = json.loads((root / "schema.json").read_text())
    schema["outputs"].update(lengths)
    schema["output_word_bits"].update({k: 32 for k in lengths})
    (root / "schema.json").write_text(json.dumps(schema) + "\n")
    (root / "primitive-scope.json").write_text(
        json.dumps(
            dict(
                scope=__doc__,
                output_narrowing="f32 probability is exported before conversion back to the unchanged half PV path; full chain remains unqualified.",
            ),
            indent=2,
        )
        + "\n"
    )
    shutil.copyfile(__file__, root / "softmax-f32-adapter.py")


if __name__ == "__main__":
    prepare(True, 3, adapt)
