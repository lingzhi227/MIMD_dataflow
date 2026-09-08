"""Conservative physical lifetimes and explicit CSL leases of the resident FFN.

In-place transforms and rotating send/receive roles are one mutable lifetime,
not permission for arbitrary overlapping SSA values. Observer/readout arrays,
compiler-managed temporaries and stack are accounted elsewhere.
"""

from inference_resources import compute, projection, registers, row_reduce
from region_lifetimes import verify


def plan(s):
    l, h, w = s["length"], s["hidden_length"], s["weight_length"]
    storage = {
        name: 2 * size
        for name, size in dict(
            x=l,
            xwork=l,
            xrecv=l,
            u=w,
            g=w,
            d=w,
            ww=w,
            wr=w,
            up=h,
            gate=h,
            hr=h,
            gamma=s["Nt"],
            rms_rows=s["Mt"],
        ).items()
    }
    values = []

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

    for name in ("x", "u", "g", "d", "gamma"):
        value("public_" + name, name, 0, 8, True)
    value("Z_to_normalized_then_rotating_X", "xwork", 0, 5)
    value("down_then_final_residual", "xwork", 7, 8)
    value("local_square_scratch", "xrecv", 0, 0)
    value("rotating_normalized_receive", "xrecv", 4, 5)
    value("row_square_sum_to_inverse", "rms_rows", 0, 3)
    value("up_to_hidden_then_rotating_send", "up", 4, 7)
    value("gate_to_silu", "gate", 5, 6)
    value("hidden_receive", "hr", 7, 7)
    for name in ("ww", "wr"):
        value("projection_" + name, name, 4, 7)
    upper = [
        i
        for i in (0, 1)
        if s["projection_stages"][i].get("accumulation") == "block_f32"
    ]
    if upper:
        storage.update(hidden_accumulator=4 * h, hidden_widened=4 * h)
        for i in upper:
            value(
                ("up" if i == 0 else "gate") + "_f32_accumulator",
                "hidden_accumulator",
                4 + i,
                4 + i,
            )
        value(
            "hidden_partial_conversion",
            "hidden_widened",
            4 + min(upper),
            4 + max(upper),
        )
    if s.get("down_accumulation") == "block_f32":
        storage.update(down_accumulator=4 * l, down_widened=4 * l)
        value("down_f32_accumulator", "down_accumulator", 7, 7)
        value("down_partial_conversion", "down_widened", 7, 7)

    def phase(name, leases):
        return dict(
            name=name, acquire=leases, release=list(leases), join_before_next=True
        )

    phases = [
        phase(
            "local_square_sum",
            dict(square=compute() + registers(2, "dest", "src0", "src1")),
        ),
        phase("row_collective", dict(row=row_reduce())),
        phase("inverse", dict(synchronous_sdk_math=[])),
        phase("normalize", dict(scale=compute())),
        phase("up_projection_complete", dict(local_compute=compute(), **projection())),
        phase(
            "gate_projection_complete", dict(local_compute=compute(), **projection())
        ),
        phase("silu_product", dict(gating=compute())),
        phase(
            "down_projection_complete", dict(local_compute=compute(), **projection())
        ),
        phase("final_residual", dict(add=compute())),
    ]
    return dict(
        storage=storage,
        values=values,
        phases=phases,
        validation=verify(storage, values, phases),
    )
