"""Callback-scoped SDK workspace leases for shared-cache attention."""

from region_lifetimes import verify
from inference_resources import registers


def storage_plan(s):
    storage = dict(s["numeric_allocations"])
    phases = []
    for i, name in enumerate(s["stages"]):
        name = f"phase{i}: " + name
        sdk = i in (1, 3, 5, 7, 9)
        bank = 1 if i in (1, 9) else 2
        lease = (
            registers(bank, "dest", "src0", "src1")
            if sdk
            else (
                registers(1, "dest", "src0", "src1") + registers(2, "dest", "src0")
                if i < 11
                else []
            )
        )
        phases.append(
            dict(
                name=name,
                acquire={name: lease},
                release=[name],
                join_before_next=True,
                completion=(
                    "SDK provider callback after broadcast and narrow"
                    if sdk
                    else "synchronous return / completed readback"
                ),
            )
        )
    values = [
        dict(
            name=k,
            storage=k,
            bytes=v,
            first=0,
            last=11,
            immutable=k in ("X", "Q", "K", "V", "W"),
        )
        for k, v in storage.items()
        if not k.startswith(("collective_", "max_"))
    ]
    for phase in (1, 5, 7, 9):
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
        values.append(dict(name=k, storage=k, bytes=storage[k], first=3, last=3))
    return dict(
        storage=storage,
        values=values,
        phases=phases,
        validation=verify(storage, values, phases),
        scope="Numeric storage and serialized DSR leases; code/SDK control/stack reserved separately, linked ELF required",
    )
