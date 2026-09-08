"""Inspect actual score/value joins and the resident projection/FFN tail."""

from frontend import check
from prefill_tail_debug import inspect as tail_inspect


def inspect(s, results, node, epoch, step):
    p = s["P"]
    offset = 2 * p + 3
    check(
        type(step) is int and 0 <= step <= offset + 4 * p + 5,
        "attention-tail debugger step",
    )
    if step >= offset:
        return tail_inspect(s, results, node, epoch, step - offset)
    out = tail_inspect(s, results, node, epoch, p)
    if step < p:
        key, stage, round_, length = (
            "score_history",
            "score_local_partial",
            step,
            s["score_length"],
        )
    elif step == p:
        key, stage, round_, length = (
            "attention_logits",
            "unscaled_qk",
            None,
            s["score_length"],
        )
    elif step == p + 1:
        key, stage, round_, length = (
            "attention_probability",
            "softmax_probability",
            None,
            s["score_length"],
        )
    elif step < 2 * p + 2:
        key, stage, round_, length = (
            "attention_value_history",
            "value_projection_prefix",
            step - p - 2,
            s["length"],
        )
    else:
        key, stage, round_, length = (
            "attention_snapshot",
            "resident_attention",
            None,
            s["length"],
        )
    observed = out["available"] and (
        round_ is None or s["instrumentation"] == "sampled"
    )
    out.update(stage=stage, round=round_, half_bits=None, observed=observed)
    for name in ("rms_progress", "prelude_progress", "f32_accumulator_bits"):
        out.pop(name, None)
    if out["available"]:
        x, y = map(int, node[1:].split("_"))
        d = results["diagnostics"][epoch]
        for name in (
            "score_progress",
            "score_roots",
            "attention_progress",
            "attention_softmax_progress",
        ):
            out[name] = d[name][y][x]
        if observed:
            out["half_bits"] = (
                d[key][y][x]
                if round_ is None
                else d[key][y][x][round_ * length : (round_ + 1) * length]
            )
    return out
