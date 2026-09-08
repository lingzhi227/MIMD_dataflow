"""Typed 16/32-bit diagnostic ports for the development resident mixed profile."""

from mesh_input_attention_sdk import extents as half_extents, packed
from mesh_mlp_sdk import WIDE_PORTS as MLP_WIDE_PORTS

WIDE_PORTS = MLP_WIDE_PORTS | {
    "wide_probability",
    "wide_exponents",
    "wide_peaks",
    "wide_sums",
    "wide_history",
    "mixed_v",
    "mixed_a",
    "mixed_projection",
    "mixed_z",
    "mixed_normalized",
    "mixed_probability_snapshot",
}


def extents(s):
    # Reuse only diagnostic extents, not the old half attention schedule/ranges.
    dimensions = {k: s[k] for k in ("P", "length", "score_length", "Mt", "instrumentation")}
    result = half_extents(dict(s, attention_schedule=dimensions))
    result.update(
        wide_probability=s["score_length"],
        wide_exponents=s["score_length"],
        wide_peaks=s["Mt"],
        wide_sums=s["Mt"],
        wide_history=5 * s["Mt"],
        mixed_probability_snapshot=s["score_length"],
    )
    result.update(
        {
            name: s["length"]
            for name in (
                "mixed_v",
                "mixed_a",
                "mixed_projection",
                "mixed_z",
                "mixed_normalized",
            )
        }
    )
    return result
