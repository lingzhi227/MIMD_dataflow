"""Saved cache-attention witnesses with physical axes; never synthesize observations."""

import re, struct
from frontend import check


def inspect(s, results, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "cache node p<column>_<row>")
    x, y = map(int, match.groups())
    stages = [
        ("score_partial", "sequence Y; partial over feature X"),
        ("score", "sequence Y; replica X"),
        ("scaled", "sequence Y; replica X"),
        ("local_max", "batch scalar; local sequence shard"),
        ("maximum", "batch scalar; global sequence maximum"),
        ("exponents", "sequence Y; replica X"),
        ("local_sum", "batch scalar; local sequence shard"),
        ("sums", "batch scalar; global denominator"),
        ("probability", "sequence Y; replica X"),
        ("context_partial", "feature X; partial over sequence Y"),
        ("context", "feature X; replica Y"),
        ("delta_partial", "feature Y; partial over input X"),
        ("delta", "feature Y; replica X"),
        ("result", "feature Y; replica X"),
    ]
    check(
        0 <= x < s["P"]
        and 0 <= y < s["P"]
        and 0 <= epoch < s["epochs"]
        and type(step) is int
        and 0 <= step < len(stages),
        "cache inspection bounds",
    )
    port, ownership = stages[step]
    length = s["numeric_allocations"][port] // 2
    available = results is not None and epoch < len(results.get("diagnostics", []))
    observed = available and (
        s["instrumentation"] == "sampled" or not port.endswith("_partial")
    )
    out = dict(
        node=node,
        epoch=epoch,
        stage=port,
        port=port,
        ownership=ownership,
        batches=s["B"],
        sequence_start=y * s["St"],
        input_feature_start=x * s["Nt"],
        output_feature_start=y * s["Nt"],
        available=available,
        observed=observed,
        length=length,
        raw_words=None,
        values=None,
        scope="Actual saved SDK words; padding scalars are not application rows; full frozen audit establishes correctness",
    )
    if available:
        d = results["diagnostics"][epoch]
        out.update(progress=d["progress"][y][x], queues=d["queues"][y][x])
        if observed:
            words = d[port][y][x]
            check(
                len(words) == length
                and all(type(v) is int and 0 <= v < 65536 for v in words),
                "cache raw half witness",
            )
            out.update(
                raw_words=words,
                values=[struct.unpack("<e", struct.pack("<H", v))[0] for v in words],
            )
    return out
