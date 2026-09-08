"""Inspect normalization, branch prefixes and completed aligned-buffer ownership."""

import re
from frontend import check
from mesh_twohop import block_index


def inspect(s, results, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "fan-out node p<column>_<row>")
    x, y = map(int, match.groups())
    p = s["P"]
    check(0 <= x < p and 0 <= y < p, "fan-out PE bounds")
    check(type(epoch) is int and 0 <= epoch < s["epochs"], "fan-out epoch bounds")
    step = 0 if step is None else step
    check(
        type(step) is int and 0 <= step <= s["projections"] * p, "fan-out prefix bounds"
    )
    out = dict(
        node=node,
        epoch=epoch,
        step=step,
        logical_rows=[y * s["Mt"], (y + 1) * s["Mt"]],
        logical_features=[x * s["Nt"], (x + 1) * s["Nt"]],
        tile_order="column-major",
        observed=s["instrumentation"] == "sampled",
    )
    length = s["Mt"] * s["Nt"]
    if step:
        branch, round = divmod(step - 1, p)
        out.update(
            branch=s["branch_bindings"][branch],
            completed_rounds=round + 1,
            consumed_k_blocks=[block_index(p, y, x, i) for i in range(round + 1)],
            boundary_live_k_block=block_index(p, y, x, 0) if round == p - 1 else None,
        )
    if results is not None:
        out["completed_calls"] = len(results["diagnostics"])
        out["available"] = epoch < out["completed_calls"]
        if not out["available"]:
            out.update(observed=False, half_bits=None)
            return out
        d = results["diagnostics"][epoch]
        out.update(
            progress=d["progress"][y][x],
            timing=d["timing"][y][x],
            queues=d["queues"][y][x],
        )
        out["half_bits"] = None
        if out["observed"]:
            out["half_bits"] = (
                d["normalized"][y][x]
                if step == 0
                else d["history"][y][x][(step - 1) * length : step * length]
            )
            if step and round == p - 1:
                out["completed_live_buffer_half_bits"] = d["reuse"][y][x][
                    branch * length : (branch + 1) * length
                ]
    return out
