"""Five joined attention phases followed by the existing eleven-phase tail."""

import copy
from prefill_tail_lifetimes import plan as tail_plan
from inference_resources import (
    compute,
    score_exchange,
    score_reduce,
    row_reduce,
    projection,
)
from region_lifetimes import verify


def plan(s):
    r = copy.deepcopy(tail_plan(s))
    storage, values = r["storage"], r["values"]
    shift = 5
    for v in values:
        if v.get("immutable"):
            v["last"] += shift
        else:
            v["first"] += shift
            v["last"] += shift
        if v["name"] == "public_x":
            v["name"] = "public_Q"
    storage.update(
        attention_k=2 * s["length"],
        attention_v=2 * s["length"],
        attention_peaks=2 * s["Mt"],
        attention_sums=2 * s["Mt"],
    )

    def value(name, allocation, first, last, size=None, immutable=False):
        values.append(
            dict(
                name=name,
                storage=allocation,
                bytes=storage[allocation] if size is None else size,
                first=first,
                last=last,
                immutable=immutable,
            )
        )

    for name in ("attention_k", "attention_v"):
        value("public_" + name, name, 0, 15, immutable=True)
    for name in ("xwork", "xrecv"):
        value("K_then_V_rotating_" + name, name, 0, 4)
    value("score_to_probability_then_rotating_left", "up", 0, 4, 2 * s["score_length"])
    value(
        "score_partial_to_exp_then_value_receive", "gate", 0, 4, 2 * s["score_length"]
    )
    value("resident_attention_output", "post_projection_z", 4, 4)
    value("softmax_local_to_global_max", "attention_peaks", 1, 2)
    value("softmax_sum_to_inverse", "attention_sums", 2, 3)

    def phase(name, leases):
        return dict(
            name=name, acquire=leases, release=list(leases), join_before_next=True
        )

    phases = [
        phase(
            "score_and_root_reductions_complete",
            dict(score_numeric=score_reduce(), **score_exchange()),
        ),
        phase(
            "softmax_maximum_complete", dict(local_max=compute(), row_max=row_reduce())
        ),
        phase(
            "softmax_exponent_sum_complete",
            dict(local_exp_sum=compute(), row_sum=row_reduce()),
        ),
        phase("softmax_normalized", dict(normalize=compute())),
        phase(
            "value_alignment_and_contraction_complete",
            dict(local_compute=compute(), **projection()),
        ),
    ] + r["phases"]
    return dict(
        storage=storage,
        values=values,
        phases=phases,
        validation=verify(storage, values, phases),
    )
