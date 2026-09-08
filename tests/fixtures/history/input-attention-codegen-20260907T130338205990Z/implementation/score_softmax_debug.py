"""Distinguish score, softmax observations and final probabilities in resident traces."""

from frontend import check
from score_debug import inspect as inspect_score


def inspect(s, results, node, epoch, step):
    p = s["P"]
    check(
        type(step) is int and 0 <= step <= p + 2, "resident score softmax debugger step"
    )
    if step < p:
        return inspect_score(s["score_schedule"], results, node, epoch, step)
    base = inspect_score(s["score_schedule"], results, node, epoch, p)
    x, y = map(int, node[1:].split("_"))
    observed = base["available"] and (
        step == p + 2 or s["instrumentation"] == "sampled"
    )
    base.update(
        stage=[
            "resident_logits",
            "softmax_statistics_and_exponents",
            "final_probability",
        ][step - p],
        observed=observed,
        half_bits=None,
    )
    if base["available"]:
        d = results["diagnostics"][epoch]
        base["softmax_progress"] = d["softmax_progress"][y][x]
        if observed:
            base["half_bits"] = d[["logits", "softmax_history", "result"][step - p]][y][
                x
            ]
            if step == p + 1:
                base["exponent_half_bits"] = d["exponents"][y][x]
    return base
