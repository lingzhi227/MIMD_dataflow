"""Source-mapped storage and explicit completion plan for projection/add/RMS."""

from inference_resources import compute, projection, registers, row_reduce
from region_lifetimes import verify


def plan(length, weight_length, rows, features):
    storage = dict(
        X_tile=2 * length,
        Q_weight_tile=2 * weight_length,
        residual_tile=2 * length,
        W_tile=2 * features,
        X_norm_tile=2 * length,
        seqLen_dim_tmp=2 * length,
        XQ_tile=2 * length,
        private_weight=2 * weight_length,
        dim_dim_tmp=2 * weight_length,
        local_sum=2 * rows,
    )
    values = []

    def value(name, allocation, first, last):
        values.append(
            dict(
                name=name,
                storage=allocation,
                bytes=storage[allocation],
                first=first,
                last=last,
            )
        )

    for name in ("X_tile", "Q_weight_tile", "residual_tile", "W_tile"):
        value("immutable_" + name, name, 0, 5)
        values[-1]["immutable"] = True
    value("projection_left_work", "X_norm_tile", 0, 0)
    value("residual_sum", "X_norm_tile", 1, 5)
    value("projection_left_receive", "seqLen_dim_tmp", 0, 0)
    value("square_scratch", "seqLen_dim_tmp", 2, 2)
    value("projection_result", "XQ_tile", 0, 1)
    value("normalized_result", "XQ_tile", 5, 5)
    value("projection_right_work", "private_weight", 0, 0)
    value("projection_right_receive", "dim_dim_tmp", 0, 0)
    # In-place local/tree sums followed by inverse are one mutable row state;
    # no generic alias exception is inferred for arbitrary operators.
    value("row_sum_then_inverse", "local_sum", 2, 5)

    def phase(name, operations):
        return dict(
            name=name,
            acquire=operations,
            release=list(operations),
            join_before_next=True,
        )

    phases = [
        phase(
            "projection_complete", dict(projection_compute=compute(), **projection())
        ),
        phase("residual_add", dict(residual_add=compute())),
        phase(
            "local_square_sum",
            dict(square_sum=compute() + registers(2, "dest", "src0", "src1")),
        ),
        phase("row_collective", dict(row_collective=row_reduce())),
        phase("inverse", dict(synchronous_sdk_math=[])),
        phase("normalize", dict(row_scale=compute())),
    ]
    return dict(
        storage=storage,
        values=values,
        phases=phases,
        validation=verify(storage, values, phases),
    )
