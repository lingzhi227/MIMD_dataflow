"""Explicit DSR leases of the pinned CSL inference library, by bank and lifetime.

These are user-visible @get_dsr reservations. Compiler-managed temporaries and
SDK launch/memcpy resources are outside this inventory. Sharing between stages
is valid only after their documented completion joins.
"""

from frontend import check


def registers(index, *banks):
    return [dict(bank=b, index=index) for b in banks]


def compute():
    return registers(1, "dest", "src0", "src1")


def projection():
    return dict(
        left_memory=registers(3, "src1", "dest"),
        right_memory=registers(4, "src1", "dest"),
        left_fabric=registers(5, "dest", "src1"),
        right_fabric=registers(6, "dest", "src1"),
    )


def score_exchange():
    return {k: v for k, v in projection().items() if k.startswith("right_")}


def row_reduce():
    return registers(2, "src1")


def score_reduce():
    return registers(1, "dest", "src0", "src1") + registers(2, "dest", "src0")


def check_disjoint(*groups):
    seen = set()
    for group in groups:
        cells = {(r["bank"], r["index"]) for r in group}
        check(not (seen & cells), "simultaneous explicit DSR lease collision")
        seen |= cells


def projection_lease():
    p = projection()
    check_disjoint(compute(), *[v for v in p.values()])
    return dict(
        compute=compute(),
        communication=p,
        completion="all four explicit microthread operations join before lease reuse",
        scope="explicit DSR reservations; compiler/SDK temporary allocation is not enumerated",
    )
