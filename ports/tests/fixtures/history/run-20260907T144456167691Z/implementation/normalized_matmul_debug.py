"""Inspect resident normalization (step0) or contraction prefixes (steps1..P)."""

import re
from frontend import check
from mesh_twohop import block_index


def inspect(s, results, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "resident node p<column>_<row>")
    x, y = map(int, match.groups())
    p = s["P"]
    check(0 <= x < p and 0 <= y < p, "resident PE bounds")
    check(type(epoch) is int and 0 <= epoch < s["epochs"], "resident epoch bounds")
    step = 0 if step is None else step
    check(
        type(step) is int and 0 <= step <= p,
        "resident step0 normalization or step1..P contraction prefix",
    )
    out = dict(
        node=node,
        epoch=epoch,
        step=step,
        operation="rmsnorm" if step == 0 else "matmul",
        logical_rows=[y * s["Mt"], (y + 1) * s["Mt"]],
        logical_features=[x * s["Nt"], (x + 1) * s["Nt"]],
        tile_order="column-major",
        observed=s["instrumentation"] == "sampled",
    )
    if step:
        out["k_blocks_consumed"] = [block_index(p, y, x, i) for i in range(step)]
    if results is not None:
        d = results["diagnostics"][epoch]
        out.update(
            progress=d["progress"][y][x],
            timing=d["timing"][y][x],
            queues=d["queues"][y][x],
        )
        if s["instrumentation"] == "sampled":
            count = s["Mt"] * s["Nt"]
            out["half_bits"] = (
                d["normalized"][y][x]
                if step == 0
                else d["history"][y][x][(step - 1) * count : step * count]
            )
        else:
            out["half_bits"] = None
        if step == p:
            out["final_half_bits"] = d["result"][y][x]
    return out
