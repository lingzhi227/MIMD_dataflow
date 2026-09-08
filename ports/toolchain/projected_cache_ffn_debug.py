"""Read actual saved parent-graph ports, including the attention/FFN boundary."""

import re, struct
from frontend import check
from projected_cache_debug import STAGES as ATTENTION_STAGES

STAGES = ATTENTION_STAGES + [
    "ffn_square_scratch",
    "ffn_rms_history",
    "ffn_sums",
    "ffn_normalized",
    "ffn_projection_partial",
    "ffn_projections",
    "ffn_activation",
    "ffn_hidden",
    "ffn_down_partial",
    "ffn_delta",
    "ffn_result",
]


def inspect(s, results, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "composed PE identifier")
    x, y = map(int, match.groups())
    check(
        0 <= x < s["P"]
        and 0 <= y < s["P"]
        and type(epoch) is int
        and 0 <= epoch < s["epochs"]
        and type(step) is int
        and 0 <= step < len(STAGES),
        "composed observation bounds",
    )
    port = STAGES[step]
    length = s["numeric_allocations"][port] // 2
    available = results is not None and epoch < len(results.get("diagnostics", []))
    out = dict(
        node=node,
        epoch=epoch,
        stage=port,
        port=port,
        available=available,
        observed=available,
        length=length,
        raw_words=None,
        values=None,
        ownership=s["ownership"],
        scope="Actual SDK observation only; unavailable calls are not synthesized from a reference.",
    )
    if available:
        raw = results["diagnostics"][epoch]
        words = raw[port][y][x]
        check(
            len(words) == length
            and all(type(v) is int and 0 <= v < 65536 for v in words),
            "composed raw half witness",
        )
        out.update(
            raw_words=words,
            values=[struct.unpack("<e", struct.pack("<H", v))[0] for v in words],
            progress=raw["progress"][y][x],
            stages=raw["stages"][y][x],
            queues=raw["queues"][y][x],
        )
    return out
