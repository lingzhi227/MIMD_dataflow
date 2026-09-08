"""Typed, observed producer joins; never fabricate half counter-mode tensors."""

import math, re, struct
from frontend import check

STAGES = [
    ("input_rms", "input_normalized", "f16"),
    ("Q_raw", "input_q_raw", "f16"),
    ("K_raw", "input_k_raw", "f16"),
    ("V", "mixed_v", "f32"),
    ("Q_rotated", "x", "f16"),
    ("K_rotated", "attention_k", "f16"),
    ("score", "attention_logits", "f16"),
    ("probability", "mixed_probability_snapshot", "f32"),
    ("attention", "mixed_a", "f32"),
    ("output_projection", "mixed_projection", "f32"),
    ("Z", "mixed_z", "f32"),
    ("normalized_Z_f32", "mixed_normalized", "f32"),
    ("normalized_Z_f16", "normalized", "f16"),
    ("up_accumulator", "up_accumulator", "f32"),
    ("gate_accumulator", "gate_accumulator", "f32"),
    ("hidden", None, "f16"),
    ("down_accumulator", "wide_accumulator", "f32"),
    ("delta", "down_snapshot", "f16"),
    ("final", "result", "f16"),
]


def inspect(s, results, node, epoch, step):
    match = re.fullmatch(r"p(\d+)_(\d+)", node)
    check(match is not None, "mixed node p<column>_<row>")
    x, y = map(int, match.groups())
    check(
        0 <= x < s["P"]
        and 0 <= y < s["P"]
        and type(epoch) is int
        and 0 <= epoch < s["epochs"]
        and type(step) is int
        and 0 <= step < len(STAGES),
        "mixed debugger bounds",
    )
    name, port, dtype = STAGES[step]
    available = results is not None and epoch < len(results.get("diagnostics", []))
    out = dict(
        node=node,
        epoch=epoch,
        stage=name,
        buffer=port,
        storage_dtype=dtype,
        word_bits=32 if dtype == "f32" else 16,
        available=available,
        observed=bool(available and port),
        raw_words=None,
        values=None,
        scope="Actual saved producer/accumulator observation; counter mode does not retain every contraction prefix or hidden tensor.",
    )
    if available:
        row = results["diagnostics"][epoch]
        out["progress"] = {
            k: row[k][y][x]
            for k in (
                "input_prefix_progress",
                "score_progress",
                "attention_progress",
                "attention_softmax_progress",
                "prelude_progress",
                "rms_progress",
                "progress",
                "queues",
            )
        }
        if port:
            words = row[port][y][x]
            out["raw_words"] = words
            check(
                all(type(v) is int and 0 <= v < 2 ** out["word_bits"] for v in words),
                "mixed raw observation word range",
            )
            values = [
                struct.unpack(
                    "<f" if dtype == "f32" else "<e",
                    struct.pack("<I" if dtype == "f32" else "<H", v),
                )[0]
                for v in words
            ]
            out["values"] = [v if math.isfinite(v) else repr(v) for v in values]
    return out
