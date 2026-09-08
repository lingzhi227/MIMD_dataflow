"""Inspect saved FFN stage words with physical feature ownership and no oracle."""

import re, struct
from frontend import check


def inspect(s, results, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "FFN node p<column>_<row>")
    x, y = map(int, match.groups())
    check(
        0 <= x < s["P"]
        and 0 <= y < s["P"]
        and 0 <= epoch < s["epochs"]
        and type(step) is int
        and 0 <= step <= 9,
        "FFN inspection bounds",
    )
    b, nt, ft = s["B"], s["Nt"], s["Ft"]
    pad = s["padded_batches"]
    stages = [
        ("local_rms", "history", 0, pad, "batch scalars"),
        ("reduced_rms", "history", pad, pad, "batch scalars"),
        ("normalized", "normalized", 0, b * nt, "Y model features"),
        (
            "local_up_gate",
            "partial",
            0,
            2 * b * ft,
            "X hidden features; partial over Y",
        ),
        (
            "reduced_up_gate",
            "projections",
            0,
            2 * b * ft,
            "X hidden features; replicated Y",
        ),
        ("activation", "activation", 0, b * ft, "X hidden features; replicated Y"),
        ("hidden_product", "hidden", 0, b * ft, "X hidden features; replicated Y"),
        ("local_down", "down_partial", 0, b * nt, "Y model features; partial over X"),
        ("reduced_down", "delta", 0, b * nt, "Y model features; replicated X"),
        ("residual", "result", 0, b * nt, "Y model features; replicated X"),
    ]
    name, port, offset, length, ownership = stages[step]
    available = results is not None and epoch < len(results.get("diagnostics", []))
    observed = available and (
        s["instrumentation"] == "sampled"
        or port not in ("history", "partial", "down_partial")
    )
    out = dict(
        node=node,
        epoch=epoch,
        stage=name,
        available=available,
        observed=observed,
        port=port,
        offset=offset,
        length=length,
        ownership=ownership,
        model_feature_start=y * nt,
        hidden_feature_start=x * ft,
        batches=b,
        raw_words=None,
        values=None,
        scope="Actual saved SDK words only; full frozen numerical audit establishes correctness. Device execution is not inferred from expected values.",
    )
    if step in (3, 4):
        out["branch_offsets"] = dict(up=0, gate=b * ft)
    if available:
        d = results["diagnostics"][epoch]
        out.update(progress=d["progress"][y][x], queues=d["queues"][y][x])
        if observed:
            words = d[port][y][x][offset : offset + length]
            check(
                len(words) == length
                and all(type(v) is int and 0 <= v < 65536 for v in words),
                "FFN raw half words",
            )
            out.update(
                raw_words=words,
                values=[struct.unpack("<e", struct.pack("<H", v))[0] for v in words],
            )
    return out
