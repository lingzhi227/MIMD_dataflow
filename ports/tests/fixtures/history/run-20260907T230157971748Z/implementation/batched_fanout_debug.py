"""Inspect actual RMS or packed branch observations, with explicit ownership."""

import re, struct
from frontend import check
from batched_rms_debug import inspect as rms_inspect


def inspect(s, results, node, epoch, step):
    if step in (0, 1, 2):
        return rms_inspect(s, results, node, epoch, step)
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "batched fanout node p<column>_<row>")
    x, y = map(int, match.groups())
    check(
        0 <= x < s["P"]
        and 0 <= y < s["P"]
        and 0 <= epoch < s["epochs"]
        and step in (3, 4),
        "batched fanout inspection bounds",
    )
    ready = results is not None and epoch < len(results.get("diagnostics", []))
    observed = ready and (step == 4 or s["instrumentation"] == "sampled")
    out = dict(
        node=node,
        epoch=epoch,
        stage=(
            "local_projection_branches" if step == 3 else "reduced_projection_branches"
        ),
        available=ready,
        observed=observed,
        raw_words=None,
        values=None,
        branch_bindings=s["branch_bindings"],
        batch_count=s["B"],
        output_feature_start=x * s["Ft"],
        output_feature_count=s["Ft"],
        input_feature_start=y * s["Nt"],
        replica_row=y,
        scope="Saved SDK words only; full frozen audit required to establish correctness.",
    )
    if ready:
        row = results["diagnostics"][epoch]
        out.update(progress=row["progress"][y][x], queues=row["queues"][y][x])
        if observed:
            words = row["partial" if step == 3 else "projections"][y][x]
            check(
                len(words) == s["padded_projection"]
                and all(type(v) is int and 0 <= v < 65536 for v in words),
                "batched fanout raw words",
            )
            out.update(
                raw_words=words,
                values=[struct.unpack("<e", struct.pack("<H", v))[0] for v in words],
            )
    return out
