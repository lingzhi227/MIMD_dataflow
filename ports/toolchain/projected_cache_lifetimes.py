"""One allocation map and callback-scoped workspace/DSR leases for the whole graph."""

from region_lifetimes import verify
from inference_resources import registers


def storage_plan(s):
    storage = dict(s["numeric_allocations"])
    phases = []
    sums = (1, 3, 5, 9, 11, 13)
    maximum = 7
    for i, name in enumerate(s["stages"]):
        sdk = i in sums or i == maximum
        if sdk:
            lease = registers(1 if i in (5, 13) else 2, "dest", "src0", "src1")
        elif i == 4:
            lease = [
                v
                for bank in range(1, 6)
                for v in registers(bank, "dest", "src0", "src1")
            ]
        elif i < 15:
            lease = registers(1, "dest", "src0", "src1") + registers(2, "dest", "src0")
        else:
            lease = []
        owner = f"phase{i}: {name}"
        phases.append(
            dict(
                name=owner,
                acquire={owner: lease},
                release=[owner],
                join_before_next=True,
                completion=(
                    "SDK provider callback after broadcast/narrow; local ownership only"
                    if sdk
                    else "Synchronous local return / completed host readback"
                ),
            )
        )
    values = [
        dict(
            name=k,
            storage=k,
            bytes=v,
            first=0,
            last=len(phases) - 1,
            immutable=k
            in ("X", "gamma", "qkv_weights", "cosine", "sine", "K", "V", "W"),
        )
        for k, v in storage.items()
        if not k.startswith(("collective_", "max_"))
    ]
    for phase in sums:
        for k in ("collective_send", "collective_reduced"):
            values.append(
                dict(
                    name=f"{k}_{phase}",
                    storage=k,
                    bytes=storage[k],
                    first=phase,
                    last=phase,
                )
            )
    for k in ("max_send", "max_gathered"):
        values.append(
            dict(name=k, storage=k, bytes=storage[k], first=maximum, last=maximum)
        )
    return dict(
        storage=storage,
        values=values,
        phases=phases,
        validation=verify(storage, values, phases),
        scope="Whole-graph nonaliased observable storage, serialized callback-scoped SDK workspace and DSR leases; linked memory still required",
    )
