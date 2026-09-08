"""Explicit FFN storage and DSR leases tied to SDK broadcast callbacks."""

from region_lifetimes import verify
from inference_resources import compute, registers
from frontend import check


def storage_plan(s):
    storage = dict(s["numeric_allocations"])
    phases = []
    # Both SDK reduce/broadcast modules retain private tasks/queues throughout.
    # Vector DSR reuse is admitted only after the entire SDK operation callback.
    descriptions = [
        ("rms_local", compute() + registers(2, "dest", "src0"), "synchronous return"),
        (
            "rms_y",
            registers(2, "dest", "src0", "src1"),
            "SDK Y broadcast callback after narrow",
        ),
        (
            "normalize_project",
            compute(),
            "synchronous normalize and two matmul returns",
        ),
        (
            "projections_y",
            registers(2, "dest", "src0", "src1"),
            "SDK Y broadcast callback after narrow",
        ),
        ("gate_down", compute(), "synchronous map/product and DOWN return"),
        (
            "down_x",
            registers(1, "dest", "src0", "src1"),
            "SDK X broadcast callback after narrow",
        ),
        ("residual", compute(), "synchronous residual then unblock"),
        ("host_readback", [], "all required D2H observations before next launch"),
    ]
    for name, leases, completion in descriptions:
        phases.append(
            dict(
                name=name,
                acquire={name: leases},
                release=[name],
                join_before_next=True,
                completion=completion,
            )
        )
    # Most exported arrays retain completed values until readback. An in-place
    # buffer is one mutable storage identity, not overlapping logical aliases.
    first = dict(
        normalized=2,
        projections=2,
        projection_partial=2,
        activation=4,
        hidden=4,
        delta=4,
        down_partial=4,
        result=6,
    )
    values = [
        dict(
            name=k,
            storage=k,
            bytes=v,
            first=first.get(k, 0),
            last=7,
            immutable=k in ("X", "gamma", "weights"),
        )
        for k, v in storage.items()
        if not k.startswith("collective_")
    ]
    # Each invocation owns both widened buffers until narrow completes. Distinct
    # logical values reuse storage only after callback, including same-plane Y/Y.
    for phase in (1, 3, 5):
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
    result = verify(storage, values, phases)
    check(
        sum(storage.values())
        + s["memory_per_pe"]["protocol_descriptors"]
        + s["memory_per_pe"]["code_stack_reserve"]
        == sum(s["memory_per_pe"].values()),
        "FFN storage ledger",
    )
    return dict(
        storage=storage,
        values=values,
        phases=phases,
        validation=result,
        scope="Declared numeric allocations and explicit synchronous/SDK DSR leases. Compiler temporaries, SDK control state and dynamic stack are reserved separately and require linked ELF/target review; local callback is not a global barrier.",
    )
