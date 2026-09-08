"""Select one PE and one row-normalization stage for humans and coding agents."""

import re
from frontend import check


def inspect(s, results, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "RMS node p<column>_<row>")
    x, y = map(int, match.groups())
    check(0 <= x < s["cols"] and 0 <= y < s["rows"], "RMS PE bounds")
    check(type(epoch) is int and 0 <= epoch < s["epochs"], "RMS epoch bounds")
    stage = 0 if step is None else step
    check(type(stage) is int and 0 <= stage < 4, "RMS stage0..3")
    out = dict(
        node=node,
        epoch=epoch,
        stage=s["stages"][stage],
        logical_rows=[y * s["Mt"], (y + 1) * s["Mt"]],
        logical_features=[x * s["Nt"], (x + 1) * s["Nt"]],
    )
    if results is not None:
        d = results["diagnostics"][epoch]
        out.update(
            progress=d["progress"][y][x],
            timing=d["timing"][y][x],
            queues=d["queues"][y][x],
        )
        if stage == 3:
            out["result_half_bits_column_major"] = d["result"][y][x]
        elif s["instrumentation"] == "sampled":
            out["row_statistic_half_bits"] = d["history"][y][x][
                stage * s["Mt"] : (stage + 1) * s["Mt"]
            ]
        else:
            out["row_statistic_half_bits"] = None
    return out
