"""Inspect the projection prelude and live residual before shared FFN stages."""

from frontend import check
from feed_forward_debug import inspect as inspect_ffn


def inspect(s, results, node, epoch, step):
    p = s["P"]
    check(type(step) is int and 0 <= step <= 4 * p + 5, "tail debugger step")
    # 0..P-1 projection prefixes; P projection; P+1 Z; P+2 normalized.
    if step >= p + 2:
        return inspect_ffn(s, results, node, epoch, step - p - 2)
    out = inspect_ffn(s, results, node, epoch, 0)
    sampled = s["instrumentation"] == "sampled"
    out.update(
        stage=(
            "output_projection_prefix"
            if step < p
            else ("output_projection" if step == p else "post_projection_residual")
        ),
        round=step if step < p else None,
        half_bits=None,
        observed=out["available"] and (sampled or step >= p),
    )
    out.pop("rms_progress", None)
    if out["available"]:
        x, y = map(int, node[1:].split("_"))
        d = results["diagnostics"][epoch]
        out["prelude_progress"] = d["prelude_progress"][y][x]
        if step < p and sampled:
            out["half_bits"] = d["projection_history"][y][x][
                step * s["length"] : (step + 1) * s["length"]
            ]
        elif step >= p:
            out["half_bits"] = d[
                "projection_snapshot" if step == p else "post_projection_z"
            ][y][x]
    return out
