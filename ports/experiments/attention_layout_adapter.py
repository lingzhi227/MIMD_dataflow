"""Explicit device-side adapter for pinned WaferLLM column-major V tiles."""


def align_column_major_value(text):
    first = text.index("fn matmul_compute()")
    last = text.index("fn rmsnorm_x()", first)
    body = text[first:last]
    needle = "right_matrix_dsd = @increment_dsd_offset(right_matrix_dsd, Nt, f16);"
    assert body.count(needle) == 1
    body = body.replace(
        needle,
        "right_matrix_dsd = @increment_dsd_offset(right_matrix_dsd, 1, f16);",
    )
    text = text[:first] + body + text[last:]
    first = text.index("fn output_matmul()")
    last = text.index("fn h1_matmul()", first)
    body = text[first:last]
    needle = "    in_preshift = true;\n    pre_remaining = offset_step;\n    shift_round = 0;\n    left_matrix_shift_callback();"
    assert body.count(needle) == 1
    body = body.replace(
        needle,
        """    right_matrix_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{dim_p_pe}->dummy[i*seq_len_p_pe]});
    value_remaining=if(px==0) 0 else if(px%2==0) P-px/2 else (px+1)/2;
    value_step=0;value_phase=true;value_shift();""",
    )
    text = text[:first] + body + text[last:]
    needle = "task right_matrix_finish() void {\n    @block(right_matrix_finish_id);"
    assert text.count(needle) == 1
    text = text.replace(needle, needle + "\n    if(value_phase){value_shift();return;}")
    text += """
var value_phase:bool=false;var value_remaining:i16=0;var value_step:i16=0;
fn value_shift() void {
 if(value_remaining>0){value_remaining-=1;swap_ptr=ptr_right_matrix_send;ptr_right_matrix_send=ptr_right_matrix_recv;ptr_right_matrix_recv=swap_ptr;comm_mod.mm_two_hop_comm_T(ptr_right_matrix_send,ptr_right_matrix_recv,value_step);value_step+=1;}
 else {value_phase=false;in_preshift=true;pre_remaining=offset_step;shift_round=0;left_matrix_shift_callback();}
}
"""
    return text
