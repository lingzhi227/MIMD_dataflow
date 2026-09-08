"""Bank-specific explicit leases for synchronous Decode-derived local stages."""

from inference_resources import registers, compute


def leases(branches):
    return dict(
        phases=[
            dict(
                stage="square_sum", registers=compute() + registers(2, "dest", "src0")
            ),
            dict(stage="rms_collective", registers=registers(2, "src1")),
            dict(stage="normalize", registers=compute()),
            dict(stage="projections", branches=branches, registers=compute()),
            dict(stage="projection_collective", registers=registers(2, "src1")),
        ],
        lifetime="Each synchronous function returns before the next phase; branch matmuls also execute sequentially. No overlapping lease or global route-quiescence inference.",
        namespace="A register is the pair (bank,index); dest2/src0-2 do not alias src1-2.",
        scope="Explicit @get_dsr reservations only. SDK memcpy/task implementation and compiler temporaries are outside this inventory.",
    )


def storage_plan(s):
    """Use the shared region verifier for actual resident numerical allocations.

    Observation keeps normalized/local/reduced state live through host readback.
    Completion tokens below mean synchronous local return, never network-wide
    quiescence. No in-place storage reuse is introduced by this metadata pass.
    """
    from frontend import check
    from region_lifetimes import verify

    b, nt, ft, count = s["B"], s["Nt"], s["Ft"], s["projections"]
    extent, padded = s["padded_projection"], s["padded_batches"]
    sampled = s["instrumentation"] == "sampled"
    storage = dict(
        X=2 * b * nt,
        W=2 * nt,
        result=2 * b * nt,
        scratch=2 * b * nt,
        sums=2 * padded,
        history=4 * padded if sampled else 2,
        weights=2 * count * nt * ft,
        projections=2 * extent,
        partial=2 * extent if sampled else 2,
    )
    descriptions = [
        ("square_sum", compute() + registers(2, "dest", "src0")),
        ("rms_collective", registers(2, "src1")),
        ("normalize", compute()),
        *[("projection_" + str(i), compute()) for i in range(count)],
        ("projection_collective", registers(2, "src1")),
        ("host_readback", []),
    ]
    phases = [
        dict(
            name=name,
            acquire={name: cells},
            release=[name],
            join_before_next=True,
            completion="Local synchronous return; no route-axis transition authorized",
        )
        for name, cells in descriptions
    ]
    last = len(phases) - 1
    values = []
    for name in storage:
        first = {"result": 2, "projections": 3, "partial": 3}.get(name, 0)
        values.append(
            dict(
                name=name,
                storage=name,
                bytes=storage[name],
                first=first,
                last=0 if name == "scratch" else last,
                immutable=name in ("X", "W", "weights"),
            )
        )
    validation = verify(storage, values, phases)
    payload = sum(
        s["memory_per_pe"][key]
        for key in (
            "data",
            "observations",
            "projection_weights",
            "projection_results",
            "projection_witnesses",
        )
    )
    # RMS data includes the separate two-byte dummy descriptor base.
    check(sum(storage.values()) + 2 == payload, "batched numerical allocation ledger")
    return dict(
        storage=storage,
        values=values,
        phases=phases,
        validation=validation,
        scope="Numerical arrays including exported witnesses; control arrays, descriptors, alignment, SDK/compiler code and dynamic stack remain separately reserved/measured. Mutable projection and RMS buffers represent successive in-place states of the same allocation.",
    )
