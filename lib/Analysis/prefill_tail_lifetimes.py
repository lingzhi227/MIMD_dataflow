"""Extend the shared FFN storage model with joined projection and a live Z."""

import copy
from feed_forward_lifetimes import plan as ff_plan
from inference_resources import compute, projection
from region_lifetimes import verify


def plan(s):
    r = copy.deepcopy(ff_plan(s))
    storage, values = r["storage"], r["values"]
    for v in values:
        if v.get("immutable"):
            v["last"] = 10
        else:
            v["first"] += 2
            v["last"] += 2
    for v in values:
        if v["name"] == "Z_to_normalized_then_rotating_X":
            v["first"] = 1
    l = s["length"]
    storage.update(
        output_weight=2 * s["output_weight_length"],
        residual=2 * l,
        post_projection_z=2 * l,
    )

    def value(name, allocation, first, last, immutable=False):
        values.append(
            dict(
                name=name,
                storage=allocation,
                bytes=storage[allocation],
                first=first,
                last=last,
                immutable=immutable,
            )
        )

    for name in ("output_weight", "residual"):
        value("public_" + name, name, 0, 10, True)
    value("projection_then_live_Z", "post_projection_z", 0, 10)
    for name in ("xwork", "xrecv", "ww", "wr"):
        value("prelude_" + name, name, 0, 0)

    def phase(name, leases):
        return dict(
            name=name, acquire=leases, release=list(leases), join_before_next=True
        )

    phases = [
        phase(
            "output_projection_complete", dict(local_compute=compute(), **projection())
        ),
        phase("post_projection_residual", dict(add=compute())),
    ] + r["phases"]
    return dict(
        storage=storage,
        values=values,
        phases=phases,
        validation=verify(storage, values, phases),
    )
