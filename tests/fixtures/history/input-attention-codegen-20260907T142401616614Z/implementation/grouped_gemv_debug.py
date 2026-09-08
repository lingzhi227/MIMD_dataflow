"""Phase-aware inspection: non-root reduction records are explicitly inactive."""

import re, struct
from frontend import check


def inspect(schedule, results, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "Grouped GEMV PE must be p<column>_<row>")
    x, y = map(int, match.groups())
    p = schedule["P"]
    check(
        0 <= x < p and 0 <= y < p and 0 <= epoch < schedule["epochs"] and 0 <= step < 3,
        "Grouped PE/epoch/phase bounds",
    )
    active = [
        True,
        y % schedule["group_size"] == schedule["root_within_group"],
        y == schedule["global_root"],
    ][step]
    output = dict(
        node=node,
        epoch=epoch,
        phase=step,
        phase_name=[
            "local contraction",
            "group-root reduction",
            "global-root reduction",
        ][step],
        phase_record_active=active,
        group=y // schedule["group_size"],
        observed=False,
    )
    if not results or epoch >= len(results.get("diagnostics", [])):
        return output
    d = results["diagnostics"][epoch]
    words = d["history_bits"][y][x][step] if d["history_bits"] is not None else None
    decode = lambda w: [struct.unpack("<e", struct.pack("<H", v))[0] for v in w]
    output.update(
        observed=True,
        phase_bits=words,
        phase_values=decode(words) if words is not None else None,
        phase_record_observed=words is not None,
        final_bits=d["result_bits"][y][x],
        final_values=decode(d["result_bits"][y][x]),
        progress=d["progress"][y][x],
        queue_empty_words=d["queue"][y][x],
        phase_record_semantics=(
            "not recorded in counters mode"
            if words is None
            else (
                "active arithmetic result"
                if active
                else "inactive local buffer snapshot, not a reduction result"
            )
        ),
    )
    return output
