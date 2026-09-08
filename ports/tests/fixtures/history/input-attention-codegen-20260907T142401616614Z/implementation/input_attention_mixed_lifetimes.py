"""Typed storage leases for the resident mixed-width continuation."""

import copy
from region_lifetimes import verify


def plan(base, s):
    r = copy.deepcopy(base)
    storage, values = r["storage"], r["values"]
    removed = {
        "local_square_scratch",
        "row_square_sum_to_inverse",
        "prelude_xwork",
        "prelude_xrecv",
        "resident_attention_output",
        "softmax_local_to_global_max",
        "softmax_sum_to_inverse",
    }
    values[:] = [v for v in values if v["name"] not in removed]
    updates = {
        "Z_to_normalized_then_rotating_X": dict(
            name="narrowed_normalized_then_rotating_X", first=19
        ),
        "projection_then_live_Z": dict(name="rounded_Z_observer", first=15),
        "computed_attention_v": dict(name="rounded_V_observer"),
        "K_then_V_rotating_xwork": dict(name="K_rotating_xwork", last=9),
        "K_then_V_rotating_xrecv": dict(name="K_rotating_xrecv", last=9),
        "score_to_probability_then_rotating_left": dict(
            name="half_score_then_probability_preview", last=12
        ),
        "score_partial_to_exp_then_value_receive": dict(
            name="half_score_partial", last=9
        ),
    }
    for v in values:
        v.update(updates.get(v["name"], {}))
    l, score, mt, nt = s["length"], s["score_length"], s["Mt"], s["Nt"]
    lengths = dict(
        wide_probability=score,
        wide_exponents=score,
        wide_peaks=mt,
        wide_sums=mt,
        wide_history=5 * mt,
        mixed_v=l,
        mixed_a=l,
        mixed_z=l,
        mixed_projection=l,
        mixed_normalized=l,
        mixed_probability_snapshot=score,
        mixed_work0=l,
        mixed_work1=l,
        mixed_left_column=mt,
        mixed_rows=mt,
        mixed_gamma=nt,
    )
    storage.update({name: 4 * size for name, size in lengths.items()})

    def add(name, allocation, first, last):
        values.append(
            dict(
                name=name,
                storage=allocation,
                bytes=storage[allocation],
                first=first,
                last=last,
                immutable=False,
            )
        )

    for name, first, last in (
        ("wide_probability", 10, 13),
        ("wide_exponents", 11, 13),
        ("wide_peaks", 10, 24),
        ("wide_sums", 11, 24),
        ("wide_history", 10, 24),
        ("mixed_v", 6, 24),
        ("mixed_a", 13, 24),
        ("mixed_z", 14, 24),
        ("mixed_projection", 14, 24),
        ("mixed_normalized", 19, 24),
        ("mixed_probability_snapshot", 12, 24),
        ("mixed_left_column", 6, 6),
        ("mixed_rows", 16, 19),
        ("mixed_gamma", 15, 19),
    ):
        add(name, name, first, last)
    for allocation, leases in {
        "mixed_work0": [
            ("V_send", 13, 13),
            ("A_send", 14, 14),
            ("residual_widen", 15, 15),
            ("delta_widen", 24, 24),
        ],
        "mixed_work1": [
            ("V_receive", 13, 13),
            ("A_receive", 14, 14),
            ("RMS_square", 16, 16),
        ],
    }.items():
        for name, first, last in leases:
            add(name, allocation, first, last)
    for v in values:
        v["storage_dtype"] = (
            "f32"
            if v["storage"] in lengths
            or v["storage"]
            in {
                "hidden_accumulator",
                "hidden_widened",
                "down_accumulator",
                "down_widened",
            }
            else "f16"
        )
    r["validation"] = verify(storage, values, r["phases"])
    r["mixed_allocation_bytes"] = sum(4 * n for n in lengths.values())
    return r


def validate(r):
    from frontend import check

    result = verify(r["storage"], r["values"], r["phases"])
    for value in r["values"]:
        allocation = value["storage"]
        expected = (
            "f32"
            if allocation.startswith(("mixed_", "wide_"))
            or allocation
            in {
                "hidden_accumulator",
                "hidden_widened",
                "down_accumulator",
                "down_widened",
            }
            else "f16"
        )
        check(
            value.get("storage_dtype") == expected,
            "logical storage dtype agrees with physical allocation",
        )
        check(
            value["bytes"] % (4 if expected == "f32" else 2) == 0,
            "typed storage byte alignment",
        )
    return result
