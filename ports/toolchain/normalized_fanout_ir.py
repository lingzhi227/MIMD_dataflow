"""Structural normalized fanout canonicalization; independent of placement."""

import copy
from frontend import check


def canonical(module):
    m = copy.deepcopy(module)
    nodes = m["nodes"]
    count = sum(n["op"] == "matmul" for n in nodes)
    check(
        count in (2, 3)
        and len(nodes) == 3 + 3 * count
        and len({n["id"] for n in nodes}) == len(nodes),
        "normalized fan-out two/three projections and unique SSA nodes",
    )
    check(
        sum(n["op"] == "rmsnorm" for n in nodes) == 1
        and sum(n["op"] == "output" for n in nodes) == count
        and sum(n["op"] == "input" for n in nodes) == count + 2,
        "normalized fan-out typed node counts",
    )
    by = {n["id"]: n for n in nodes}
    norm = next(n for n in nodes if n["op"] == "rmsnorm")
    check(
        len(norm["inputs"]) == 2 and all(v in by for n in nodes for v in n["inputs"]),
        "normalized fan-out defined dependencies",
    )
    x, w = [by[v] for v in norm["inputs"]]
    ordered = [x, w, norm]
    consumed = []
    for out in [n for n in nodes if n["op"] == "output"]:
        check(len(out["inputs"]) == 1, "normalized fan-out output edge")
        mm = by[out["inputs"][0]]
        check(
            mm["op"] == "matmul"
            and len(mm["inputs"]) == 2
            and mm["inputs"][0] == norm["id"],
            "normalized fan-out shared producer",
        )
        q = by[mm["inputs"][1]]
        check(q["op"] == "input", "normalized fan-out weight input")
        ordered.extend([q, mm, out])
        consumed.extend([q["id"], mm["id"], out["id"]])
    check(
        len(set(consumed + [x["id"], w["id"], norm["id"]])) == len(nodes),
        "normalized fan-out distinct branch state and weights",
    )
    check(
        len({n["host"] for n in ordered if n["op"] == "input"}) == count + 2
        and len({n["host"] for n in ordered if n["op"] == "output"}) == count,
        "normalized fan-out unique host ports",
    )
    m["nodes"] = ordered
    return m
