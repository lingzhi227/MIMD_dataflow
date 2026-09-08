"""Inspect logical blocks through both-axis alignment and overlapped contraction."""

import re
from frontend import check
from mesh_twohop import block_index


def inspect(s, results, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "device matmul node p<column>_<row>")
    x, y = map(int, match.groups())
    p = s["P"]
    check(
        0 <= x < p
        and 0 <= y < p
        and type(epoch) is int
        and 0 <= epoch < s["epochs"]
        and type(step) is int
        and 0 <= step <= p,
        "device matmul debugger bounds",
    )
    available = results is not None and epoch < len(results.get("diagnostics", []))
    observed = available and (step == p or s["instrumentation"] == "sampled")
    out = dict(
        node=node,
        epoch=epoch,
        round=step,
        stage=(
            "final_product"
            if step == p
            else "contraction_prefix_after_device_alignment"
        ),
        k_block=block_index(p, y, x, step) if step < p else None,
        rhs_stride=s["Mt"],
        available=available,
        observed=observed,
        half_bits=None,
        left_owner_half_bits=None,
        right_owner_half_bits=None,
    )
    if available:
        d = results["diagnostics"][epoch]
        out.update(progress=d["progress"][y][x], timing=d["timing"][y][x])
        if observed:
            out["half_bits"] = (
                d["result"][y][x]
                if step == p
                else d["history"][y][x][step * s["length"] : (step + 1) * s["length"]]
            )
            if step < p:
                out["left_owner_half_bits"] = d["left_owners"][y][x][
                    step * s["left_length"] : (step + 1) * s["left_length"]
                ]
                out["right_owner_half_bits"] = d["right_owners"][y][x][
                    step * s["length"] : (step + 1) * s["length"]
                ]
    return out
