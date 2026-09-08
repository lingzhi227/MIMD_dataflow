"""Observe resident normalization, shared MLP phases, delta and final residual."""

from frontend import check
from mlp_debug import inspect as inspect_mlp


def inspect(s, results, node, epoch, step):
    p = s["P"]
    check(type(step) is int and 0 <= step <= 3 * p + 3, "FFN debugger step")
    # Reuse the shared engine's coordinate/epoch checks and protocol evidence.
    inner_step = min(max(step - 1, 0), 3 * p + 1)
    out = inspect_mlp(s, results, node, epoch, inner_step)
    if 1 <= step <= 3 * p + 1:
        return out
    stage, key = (
        ("normalized", "normalized")
        if step == 0
        else (
            ("delta", "down_snapshot")
            if step == 3 * p + 2
            else ("residual_output", "result")
        )
    )
    out.update(
        stage=stage, round=None, k_block=None, observed=out["available"], half_bits=None
    )
    out.pop("f32_accumulator_bits", None)
    if out["available"]:
        x, y = map(int, node[1:].split("_"))
        d = results["diagnostics"][epoch]
        out["half_bits"] = d[key][y][x]
        out["rms_progress"] = d["rms_progress"][y][x]
        if stage == "delta":
            out["f32_accumulator_bits"] = d["wide_accumulator"][y][x]
    return out
