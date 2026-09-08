"""Whole-composition ownership; observations remain live until final readback."""

import copy
from frontend import check
from inference_resources import registers
from region_lifetimes import verify


def storage_plan(schedule):
    prefix = schedule["attention"]["storage_lifetimes"]
    phases = copy.deepcopy(prefix["phases"][:-1])
    additions = [
        ("ffn_square_sum", None),
        ("ffn_mean_y", 2),
        ("ffn_normalize_up_gate", None),
        ("ffn_projections_y", 2),
        ("ffn_silu_product_down", None),
        ("ffn_down_x", 1),
        ("ffn_residual_z", None),
        ("host_readback", 0),
    ]
    for name, bank in additions:
        lease = (
            []
            if bank == 0
            else (
                registers(bank, "dest", "src0", "src1")
                if bank
                else registers(1, "dest", "src0", "src1") + registers(2, "dest", "src0")
            )
        )
        phases.append(
            dict(
                name=name,
                acquire={name: lease},
                release=[name],
                join_before_next=True,
                completion=(
                    "SDK broadcast callback after narrow; local ownership only"
                    if bank
                    else "synchronous local completion or final host readback"
                ),
            )
        )
    storage = dict(schedule["numeric_allocations"])
    last = len(phases) - 1
    check(last == 22, "composed callback schedule phase count")
    values = []
    for name, size in storage.items():
        if name.startswith(("collective_", "max_", "mean_")):
            continue
        values.append(
            dict(
                name=name,
                storage=name,
                bytes=size,
                first=0,
                last=last,
                immutable=name
                in (
                    "X",
                    "gamma",
                    "qkv_weights",
                    "cosine",
                    "sine",
                    "K",
                    "V",
                    "W",
                    "ffn_weights",
                ),
            )
        )
    for phase in (1, 3, 5, 9, 11, 13, 18, 20):
        for name in ("collective_send", "collective_reduced"):
            values.append(
                dict(
                    name=f"{name}_{phase}",
                    storage=name,
                    bytes=storage[name],
                    first=phase,
                    last=phase,
                )
            )
    for phase, prefix_name in ((7, "max"), (16, "mean")):
        for name in (
            ("max_send", "max_gathered")
            if prefix_name == "max"
            else ("mean_send", "mean_reduced")
        ):
            values.append(
                dict(
                    name=f"{name}_{phase}",
                    storage=name,
                    bytes=storage[name],
                    first=phase,
                    last=phase,
                )
            )
    return dict(
        storage=storage,
        values=values,
        phases=phases,
        validation=verify(storage, values, phases),
        scope="Declared callback/DSR and numeric-storage leases; does not prove dynamic stack or cross-PE global barriers. No observation lifetime shortened.",
    )
