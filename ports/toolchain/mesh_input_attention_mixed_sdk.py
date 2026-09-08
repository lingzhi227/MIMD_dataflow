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
    dimensions = {
        k: s[k] for k in ("P", "length", "score_length", "Mt", "instrumentation")
    }
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


def run(root):
    from half_region_runtime import run as execute, read
    from mesh_mlp_sdk import parameters
    from mesh_feed_forward_sdk import decode

    execute(
        root,
        parameters,
        extents,
        packed,
        decode,
        word_bits={
            k: 32 for k in extents(read(root, "schedule.json")) if k in WIDE_PORTS
        },
    )


def audit(root):
    from half_region_runtime import read
    from input_attention_mixed_audit import audit_cases

    return audit_cases(
        read(root, "schedule.json"),
        read(root, "semantic.json"),
        read(root, "batches.json"),
        read(root, "results.json"),
    )
