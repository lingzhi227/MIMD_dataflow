"""Observed projection prefixes, residual sum and final row scales."""

import re
from frontend import check
from mesh_twohop import block_index


def inspect(s, results, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "node p<column>_<row>")
    x, y = map(int, match.groups())
    p = s["P"]
    check(
        0 <= x < p
        and 0 <= y < p
        and type(epoch) is int
        and 0 <= epoch < s["epochs"]
        and type(step) is int
        and 0 <= step <= p + 1,
        "composition debugger bounds",
    )
    phase = (
        "projection"
        if step < p
        else "residual_sum" if step == p else "normalized_output"
    )
    key = "history" if step < p else "sum" if step == p else "result"
    available = results is not None and epoch < len(results.get("diagnostics", []))
    observed = available and (step == p + 1 or s["instrumentation"] == "sampled")
    out = dict(
        node=node,
        epoch=epoch,
        stage=phase,
        round=step if step < p else None,
        k_block=block_index(p, y, x, step) if step < p else None,
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
        if observed:
            offset = step * s["length"] if step < p else 0
            out["half_bits"] = d[key][y][x][offset : offset + s["length"]]
        if step == p + 1:
            out["row_inverse_half_bits"] = d["inverse"][y][x]
    return out
