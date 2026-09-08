"""Actual whole-graph saved observations; no expected-state substitution."""

import re, struct
from frontend import check

STAGES = [
    "rms_scratch",
    "rms_history",
    "rms_sums",
    "normalized",
    "projection_partial",
    "projections",
    "query_pair_history",
    "Q",
    "key_pair_history",
    "rotated_key",
    "pair_scratch",
    "score_partial",
    "score",
    "scaled",
    "local_max",
    "maximum",
    "exponents",
    "local_sum",
    "sums",
    "probability",
    "context_partial",
    "context",
    "delta_partial",
    "delta",
    "result",
]


def inspect(s, results, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "projected cache node p<column>_<row>")
    x, y = map(int, match.groups())
    check(
        0 <= x < s["P"]
        and 0 <= y < s["P"]
        and type(epoch) is int
        and 0 <= epoch < s["epochs"]
        and type(step) is int
        and 0 <= step < len(STAGES),
        "projected cache inspection bounds",
    )
    port = STAGES[step]
    available = results is not None and epoch < len(results.get("diagnostics", []))
    omitted = port.endswith("_partial") or port.endswith("_history")
    observed = available and (s["instrumentation"] == "sampled" or not omitted)
    length = s["numeric_allocations"][port] // 2
    out = dict(
        node=node,
        epoch=epoch,
        stage=port,
        port=port,
        available=available,
        observed=observed,
        length=length,
        raw_words=None,
        values=None,
        ownership=s["ownership"],
        scope="Actual saved all-graph SDK words; scalar padding is not an application row; pair scratch contains only the last K batch products",
    )
    if available:
        d = results["diagnostics"][epoch]
        out.update(progress=d["progress"][y][x], queues=d["queues"][y][x])
        if observed:
            words = d[port][y][x]
            check(
                len(words) == length
                and all(type(v) is int and 0 <= v < 65536 for v in words),
                "projected cache raw half witness",
            )
            out.update(
                raw_words=words,
                values=[struct.unpack("<e", struct.pack("<H", v))[0] for v in words],
            )
    return out
