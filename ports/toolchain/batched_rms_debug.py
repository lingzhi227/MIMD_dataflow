"""Actual batch-major producer observations with explicit padding and replicas."""

import re, struct
from frontend import check


def inspect(s, results, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "batched RMS node p<column>_<row>")
    x, y = map(int, match.groups())
    check(
        0 <= x < s["P"]
        and 0 <= y < s["P"]
        and 0 <= epoch < s["epochs"]
        and step in (0, 1, 2),
        "batched RMS inspection bounds",
    )
    ready = results is not None and epoch < len(results.get("diagnostics", []))
    observed = ready and (step != 0 or s["instrumentation"] == "sampled")
    result = dict(
        node=node,
        epoch=epoch,
        stage=("local_square_sum", "grouped_sum", "normalized")[step],
        storage_dtype="f16",
        available=ready,
        observed=observed,
        raw_words=None,
        values=None,
        feature_start=y * s["Nt"],
        feature_count=s["Nt"],
        batch_count=s["B"],
        replica_column=x,
        padded_slots=list(range(s["B"], s["padded_batches"])) if step != 2 else [],
        scope="Raw saved device observations; values alone do not imply validated execution or mesh-wide quiescence.",
    )
    if ready:
        row = results["diagnostics"][epoch]
        result["progress"] = row["progress"][y][x]
        result["queues"] = row["queues"][y][x]
        if observed:
            words = (
                row["history"][y][x][: s["padded_batches"]]
                if step == 0
                else row["sums" if step == 1 else "result"][y][x]
            )
            check(all(type(v) is int and 0 <= v < 65536 for v in words), "half words")
            result["raw_words"] = words
            result["values"] = [
                struct.unpack("<e", struct.pack("<H", v))[0] for v in words
            ]
    return result
