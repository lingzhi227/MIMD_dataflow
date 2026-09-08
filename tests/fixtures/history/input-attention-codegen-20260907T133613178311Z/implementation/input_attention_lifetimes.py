"""Nine prefix phases followed by the sixteen-phase resident attention tail."""

import copy
from attention_tail_lifetimes import plan as attention_lifetimes
from inference_resources import compute, registers, row_reduce, projection
from region_lifetimes import verify


def plan(s):
    r = copy.deepcopy(attention_lifetimes(s))
    storage, values = r["storage"], r["values"]
    shift = 9
    last = len(r["phases"]) + shift - 1
    for value in values:
        if value.get("immutable"):
            value["last"] += shift
        else:
            value["first"] += shift
            value["last"] += shift
        computed = {"public_Q": 4, "public_attention_k": 5, "public_attention_v": 6}
        if value["name"] in computed:
            value.update(
                first=computed[value["name"]],
                last=last,
                immutable=False,
                name="computed_" + value["name"][7:],
            )
        elif value["name"] == "public_residual":
            value["name"] = "public_original_X"
    storage.update(
        {
            name: 2 * s["output_weight_length"]
            for name in ("q_weight", "k_weight", "v_weight")
        }
    )
    storage.update(cosine=s["Nt"], sine=s["Nt"])

    def add(name, allocation, first, last, size=None, immutable=False):
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

    for name in ("q_weight", "k_weight", "v_weight", "cosine", "sine"):
        add("public_" + name, name, 0, last, immutable=True)
    add("input_normalized_then_shared_rotating_owner", "xwork", 0, 6)
    add("input_square_scratch", "xrecv", 0, 0)
    add("input_normalized_receive", "xrecv", 4, 6)
    add("input_row_sum_to_inverse", "rms_rows", 0, 3)
    for name in ("ww", "wr"):
        add("prefix_projection_" + name, name, 4, 6)
    add("Q_then_K_pair_scratch", "xwork", 7, 8, 8 * s["Mt"])

    def phase(name, leases):
        return dict(
            name=name, acquire=leases, release=list(leases), join_before_next=True
        )

    phases = [
        phase(
            "input_square_sum",
            dict(square=compute() + registers(2, "dest", "src0", "src1")),
        ),
        phase("input_row_reduce", dict(row=row_reduce())),
        phase("input_inverse", dict(synchronous_sdk_math=[])),
        phase("input_normalize", dict(scale=compute())),
    ]
    phases.extend(
        phase(
            name + "_projection_complete", dict(local_compute=compute(), **projection())
        )
        for name in ("Q", "K", "V")
    )
    phases.extend(
        phase(name + "_pair_transform_complete", dict(synchronous_memory_dsds=[]))
        for name in ("Q", "K")
    )
    phases += r["phases"]
    return dict(
        storage=storage,
        values=values,
        phases=phases,
        validation=verify(storage, values, phases),
    )
