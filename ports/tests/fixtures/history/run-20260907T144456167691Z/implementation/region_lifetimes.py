"""Validate declared stage lifetimes and explicit resource acquire/release plans.

This checks scheduled IR, not arbitrary CSL control flow. A lowering must tie
its completion contracts to executed callbacks and audit actual target state.
Unmodelled compiler/SDK resources and diagnostic arrays are outside this pass.
"""

from frontend import check


def verify(storage, values, phases):
    check(
        bool(phases) and len({p["name"] for p in phases}) == len(phases),
        "unique region phases",
    )
    check(
        all(type(size) is int and size > 0 for size in storage.values()),
        "positive physical storage extents",
    )
    check(len({v["name"] for v in values}) == len(values), "unique logical lifetimes")
    for value in values:
        start, end = value["first"], value["last"]
        check(
            type(start) is int and type(end) is int and 0 <= start <= end < len(phases),
            "region lifetime endpoints",
        )
        check(
            value["storage"] in storage
            and type(value["bytes"]) is int
            and 0 < value["bytes"] <= storage[value["storage"]],
            "logical value fits physical storage",
        )
        if value.get("immutable", False):
            check(
                start == 0 and end == len(phases) - 1,
                "immutable public storage survives the complete region",
            )
    for i, left in enumerate(values):
        for right in values[i + 1 :]:
            if left["storage"] == right["storage"]:
                check(
                    left["last"] < right["first"] or right["last"] < left["first"],
                    "simultaneously live values alias one storage object",
                )
    active = {}
    for phase in phases:
        for token, leases in phase["acquire"].items():
            check(
                token not in active and isinstance(token, str) and bool(token),
                "unique active completion token",
            )
            cells = [(v["bank"], v["index"]) for v in leases]
            check(
                len(cells) == len(set(cells)),
                "duplicate explicit resource in one lease",
            )
            check(
                all(
                    isinstance(bank, str) and type(index) is int and index >= 0
                    for bank, index in cells
                ),
                "well-formed explicit resource",
            )
            occupied = set().union(*active.values()) if active else set()
            check(
                not set(cells) & occupied, "explicit resource reused before completion"
            )
            active[token] = set(cells)
        for token in phase["release"]:
            check(token in active, "completion must match an active token")
            del active[token]
        if phase.get("join_before_next", False):
            check(not active, "phase join leaves unfinished operations")
    check(not active, "region exits with unfinished operations")
    return dict(
        checked=True,
        numerical_storage_bytes=sum(storage.values()),
        logical_values=len(values),
        phases=len(phases),
        scope="Declared numerical storage and explicit leases only; lowering callbacks, dynamic stack, SDK/compiler temporaries and diagnostic storage require separate target validation",
    )
