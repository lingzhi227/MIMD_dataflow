"""Structural RMS/UP/GATE/SiLU/product/DOWN/residual graph, independent of layout."""

import copy
from frontend import check


def canonical(module):
    m = copy.deepcopy(module)
    ns = m["nodes"]
    by = {n["id"]: n for n in ns}
    check(
        len(by) == len(ns) == 13
        and sorted(n["op"] for n in ns)
        == sorted(
            ["input"] * 5
            + ["matmul"] * 3
            + ["rmsnorm", "silu", "multiply", "add", "output"]
        ),
        "feed-forward thirteen unique typed nodes",
    )
    check(all(i in by for n in ns for i in n["inputs"]), "defined feed-forward edges")

    def operands(n, op, count):
        check(
            n["op"] == op and len(n["inputs"]) == count, "feed-forward " + op + " edges"
        )
        return [by[i] for i in n["inputs"]]

    out = next(n for n in ns if n["op"] == "output")
    (add,) = operands(out, "output", 1)
    pair = operands(add, "add", 2)
    check(
        sum(n["op"] == "matmul" for n in pair) == 1,
        "final residual consumes the down projection",
    )
    down = next(n for n in pair if n["op"] == "matmul")
    z = next(n for n in pair if n is not down)
    hidden, wd = operands(down, "matmul", 2)
    pair = operands(hidden, "multiply", 2)
    check(sum(n["op"] == "silu" for n in pair) == 1, "feed-forward gating")
    act = next(n for n in pair if n["op"] == "silu")
    up = next(n for n in pair if n is not act)
    (gate,) = operands(act, "silu", 1)
    norm, wu = operands(up, "matmul", 2)
    norm2, wg = operands(gate, "matmul", 2)
    zz, gamma = operands(norm, "rmsnorm", 2)
    check(
        norm["id"] == norm2["id"] and z["id"] == zz["id"],
        "RMS input is the final residual; shared normalized activation",
    )
    sources = [z, gamma, wu, wg, wd]
    check(
        all(n["op"] == "input" and not n["inputs"] for n in sources)
        and len({n["id"] for n in sources}) == len({n["host"] for n in sources}) == 5,
        "five distinct feed-forward inputs",
    )
    m["nodes"] = sources + [norm, up, gate, act, hidden, down, add, out]
    return m
