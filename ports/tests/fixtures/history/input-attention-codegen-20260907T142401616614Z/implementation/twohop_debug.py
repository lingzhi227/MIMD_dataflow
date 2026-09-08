"""Decode a selected two-hop PE/round without importing the SDK runtime."""

from mesh_twohop import block_index
from frontend import check
import re, struct


def inspect(schedule, results, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "Two-hop PE must be p<column>_<row>")
    x, y = map(int, match.groups())
    p = schedule["P"]
    check(
        0 <= x < p and 0 <= y < p and 0 <= epoch < schedule["epochs"] and 0 <= step < p,
        "Two-hop PE/epoch/round bounds",
    )
    out = dict(
        node=node,
        epoch=epoch,
        round=step,
        k_block=block_index(p, y, x, step),
        observed=False,
        tile_order="column-major",
        accumulation="binary16 fused multiply-add",
        resources=schedule["resources"],
    )
    if not results or epoch >= len(results.get("diagnostics", [])):
        return out
    d = results["diagnostics"][epoch]
    decode = lambda words: [struct.unpack("<e", struct.pack("<H", w))[0] for w in words]
    history = d["history_bits"][y][x][step] if d["history_bits"] is not None else None
    witnesses = d["witness_bits"][y][x][step]
    t = d["timing"][y][x][step]
    out.update(
        observed=True,
        prefix_observed=history is not None,
        prefix_bits=history,
        prefix_values=decode(history) if history is not None else None,
        operand_corner_bits=witnesses,
        operand_corner_values=decode(witnesses),
        operand_corner_order=["X_first", "X_last", "W_first", "W_last"],
        progress=dict(
            zip(
                [
                    "compute_rounds",
                    "exchange_rounds",
                    "x_callbacks",
                    "w_callbacks",
                    "alignment_calls",
                    "alignment_callbacks",
                    "warm_entries",
                ],
                d["progress"][y][x],
            )
        ),
        queue_empty_words=d["queue"][y][x],
        compute_cycles=sum((t[i + 3] - t[i]) << (16 * i) for i in range(3)),
    )
    return out
