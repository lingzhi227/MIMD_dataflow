"""Inspect physical MLP phases without inventing unobserved counter tensors."""

import re
from frontend import check
from mesh_twohop import block_index


def inspect(s, results, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "MLP node p<column>_<row>")
    x, y = map(int, match.groups())
    p = s["P"]
    check(
        0 <= x < p
        and 0 <= y < p
        and type(epoch) is int
        and 0 <= epoch < s["epochs"]
        and type(step) is int
        and 0 <= step <= 3 * p + 1,
        "MLP debugger bounds",
    )
    if step < p:
        phase = "up"
        r = step
        key = "up_history"
        length = s["hidden_length"]
    elif step < 2 * p:
        phase = "gate"
        r = step - p
        key = "gate_history"
        length = s["hidden_length"]
    elif step == 2 * p:
        phase = "hidden"
        r = None
        key = "hidden_snapshot"
        length = s["hidden_length"]
    elif step <= 3 * p:
        phase = "down"
        r = step - 2 * p - 1
        key = "down_history"
        length = s["length"]
    else:
        phase = "output"
        r = None
        key = "result"
        length = s["length"]
    available = results is not None and epoch < len(results.get("diagnostics", []))
    observed = available and (phase == "output" or s["instrumentation"] == "sampled")
    out = dict(
        node=node,
        epoch=epoch,
        stage=phase,
        round=r,
        k_block=block_index(p, y, x, r) if r is not None else None,
        available=available,
        observed=observed,
        half_bits=None,
    )
    if available:
        d = results["diagnostics"][epoch]
        out.update(
            progress=d["progress"][y][x],
            timing=d["timing"][y][x],
            queues=d["queues"][y][x],
        )
        if phase == "output" and s.get("down_accumulation") == "block_f32":
            out["f32_accumulator_bits"] = d["wide_accumulator"][y][x]
        if phase in ("up", "gate") and r == p - 1 and phase + "_accumulator" in d:
            out["f32_accumulator_bits"] = d[phase + "_accumulator"][y][x]
        if observed:
            out["half_bits"] = d[key][y][x][(r or 0) * length : ((r or 0) + 1) * length]
    return out
