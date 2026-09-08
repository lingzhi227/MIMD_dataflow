"""Inspect a score round, physical K owner and rotating reduction root."""

import re
from frontend import check
from mesh_twohop import cycle


def inspect(s, results, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "score node p<column>_<row>")
    x, y = map(int, match.groups())
    p = s["P"]
    check(
        0 <= x < p
        and 0 <= y < p
        and type(epoch) is int
        and 0 <= epoch < s["epochs"]
        and type(step) is int
        and 0 <= step <= p,
        "score debugger bounds",
    )
    order = cycle(p)
    root = order[(order.index(y) - step) % p] if step < p else None
    available = results is not None and epoch < len(results.get("diagnostics", []))
    observed = available and (step == p or s["instrumentation"] == "sampled")
    out = dict(
        node=node,
        epoch=epoch,
        round=step,
        stage="final_score" if step == p else "partial_before_root_reduction",
        logical_query_rows=[y * s["Mt"], (y + 1) * s["Mt"]],
        feature_partition=x,
        k_token_block=root,
        reduction_root=root,
        available=available,
        observed=observed,
        half_bits=None,
        owner_half_bits=None,
    )
    if available:
        d = results["diagnostics"][epoch]
        out.update(
            progress=d["progress"][y][x],
            timing=d["timing"][y][x],
            roots=d["roots"][y][x],
        )
        if observed:
            out["half_bits"] = (
                d["result"][y][x]
                if step == p
                else d["history"][y][x][
                    step * s["score_length"] : (step + 1) * s["score_length"]
                ]
            )
            if step < p:
                out["owner_half_bits"] = d["owners"][y][x][
                    step * s["length"] : (step + 1) * s["length"]
                ]
    return out
