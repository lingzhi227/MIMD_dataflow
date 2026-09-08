"""Inspect local four-product history or final pair rotation with explicit observation scope."""

import re
from frontend import check


def inspect(s, results, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "pair rotation node p<column>_<row>")
    x, y = map(int, match.groups())
    check(0 <= x < s["cols"] and 0 <= y < s["rows"], "pair rotation PE bounds")
    check(type(epoch) is int and 0 <= epoch < s["epochs"], "pair rotation epoch bounds")
    stage = 0 if step is None else step
    check(type(stage) is int and stage in (0, 1), "pair rotation stage0/1")
    out = dict(
        node=node,
        epoch=epoch,
        stage=s["stages"][stage],
        logical_rows=[y * s["Mt"], (y + 1) * s["Mt"]],
        logical_features=[x * s["Nt"], (x + 1) * s["Nt"]],
        tile_order="column-major",
        observed=stage == 1 or s["instrumentation"] == "sampled",
    )
    if results is not None:
        d = results["diagnostics"][epoch]
        out.update(progress=d["progress"][y][x], timing=d["timing"][y][x])
        out["half_bits"] = (
            d["result" if stage else "history"][y][x] if out["observed"] else None
        )
    return out
