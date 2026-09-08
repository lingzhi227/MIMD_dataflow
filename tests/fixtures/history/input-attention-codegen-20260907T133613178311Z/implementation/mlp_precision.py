"""Optional block-f32 up/gate precision around the shared projection engine."""


def hooks(s):
    enabled = [
        i
        for i in (0, 1)
        if s["projection_stages"][i].get("accumulation") == "block_f32"
    ]
    if not enabled:
        return {}
    condition = " or ".join(f"phase=={i}" for i in enabled)
    declarations = """
const hidden_accum=@import_module("block_accumulate.csl",.{.length=H});
const hidden_words_dsd=@get_dsd(mem1d_dsd,.{.base_address=hidden_accum.words,.extent=@as(u16,H)});
"""
    result = dict(
        PARTIAL_RESET=f"if({condition}){{@fmovh(@set_dsd_length(@set_dsd_base_addr(hv,ptr_out_matrix),@as(u16,Mt*Nt)),0.0);}}",
        PARTIAL_MERGE=f"if({condition}){{hidden_accum.merge(ptr_out_matrix);}}",
        ACCUM_RESET=f"if({condition}){{hidden_accum.reset();}}",
    )
    for i in enabled:
        name = "up" if i == 0 else "gate"
        declarations += f"""
var {name}_words=@zeros([H]u32);
const {name}_words_dsd=@get_dsd(mem1d_dsd,.{{.tensor_access=|j|{{H}}->{name}_words[j]}});
var {name}_words_ptr:[*]u32=&{name}_words;
comptime {{@export_symbol({name}_words_ptr,"{name}_accumulator");}}
"""
        result[name.upper() + "_FINISH"] = (
            f"hidden_accum.snapshot();@mov32({name}_words_dsd,hidden_words_dsd);"
        )
    result["DECLARATIONS"] = declarations
    return result
