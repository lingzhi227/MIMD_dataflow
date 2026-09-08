"""Structural supplied Q/K/V attention graph, independent of spatial placement."""

import copy
from frontend import check


def canonical(module):
    m = copy.deepcopy(module)
    by = {n["id"]: n for n in m["nodes"]}
    check(
        len(by) == len(m["nodes"]) == 8
        and sorted(n["op"] for n in m["nodes"])
        == [
            "input",
            "input",
            "input",
            "matmul",
            "matmul",
            "output",
            "softmax",
            "transpose",
        ],
        "resident attention eight unique typed nodes",
    )
    check(
        all(i in by for n in m["nodes"] for i in n["inputs"]),
        "resident attention defined operands",
    )
    out = next(n for n in m["nodes"] if n["op"] == "output")
    check(len(out["inputs"]) == 1, "attention output edge")
    pv = by[out["inputs"][0]]
    check(
        pv["op"] == "matmul" and len(pv["inputs"]) == 2, "attention value contraction"
    )
    sm, v = [by[i] for i in pv["inputs"]]
    check(
        sm["op"] == "softmax" and len(sm["inputs"]) == 1 and v["op"] == "input",
        "attention resident probability edge",
    )
    mm = by[sm["inputs"][0]]
    check(
        mm["op"] == "matmul" and len(mm["inputs"]) == 2, "attention score contraction"
    )
    q, t = [by[i] for i in mm["inputs"]]
    check(
        t["op"] == "transpose" and len(t["inputs"]) == 1, "attention key transpose view"
    )
    k = by[t["inputs"][0]]
    check(
        q["op"] == k["op"] == "input" and len({q["id"], k["id"], v["id"]}) == 3,
        "attention distinct supplied inputs",
    )
    m["nodes"] = [q, k, v, t, mm, sm, pv, out]
    return m
