"""Structural normalized three-projection/pair/cache graph, independent of names."""

import copy
from frontend import check


def canonical(module):
    m = copy.deepcopy(module)
    nodes = m["nodes"]
    by = {n["id"]: n for n in nodes}
    check(
        len(nodes) == len(by) == 25
        and sorted(n["op"] for n in nodes)
        == sorted(
            ["input"] * 10
            + ["matmul"] * 6
            + ["rotate_pairs"] * 2
            + ["rmsnorm", "transpose", "softmax", "add"]
            + ["output"] * 3
        ),
        "projected cache twenty-five unique typed nodes",
    )
    check(
        all(i in by for n in nodes for i in n["inputs"]),
        "projected cache defined edges",
    )

    def args(n, op, count):
        check(
            n["op"] == op and len(n["inputs"]) == count,
            "projected cache " + op + " edges",
        )
        return [by[i] for i in n["inputs"]]

    outs = [n for n in nodes if n["op"] == "output"]
    check(
        all(type(n.get("host")) is str and n["host"] for n in outs)
        and len({n["host"] for n in outs}) == 3,
        "projected cache three distinct output ports",
    )
    ends = [args(n, "output", 1)[0] for n in outs]
    check(
        sorted(n["op"] for n in ends) == ["add", "matmul", "rotate_pairs"],
        "projected cache three semantic outputs",
    )
    add = next(n for n in ends if n["op"] == "add")
    pair = args(add, "add", 2)
    check(
        sorted(n["op"] for n in pair) == ["input", "matmul"],
        "original-input residual and output projection",
    )
    x = next(n for n in pair if n["op"] == "input")
    delta = next(n for n in pair if n["op"] == "matmul")
    context, wo = args(delta, "matmul", 2)
    prob, value = args(context, "matmul", 2)
    (score,) = args(prob, "softmax", 1)
    rq, kt = args(score, "matmul", 2)
    (key,) = args(kt, "transpose", 1)
    q, c, s = args(rq, "rotate_pairs", 3)
    rk = next(n for n in ends if n["op"] == "rotate_pairs")
    k, cc, ss = args(rk, "rotate_pairs", 3)
    v = next(n for n in ends if n["op"] == "matmul")
    norm, wq = args(q, "matmul", 2)
    nk, wk = args(k, "matmul", 2)
    nv, wv = args(v, "matmul", 2)
    xx, gamma = args(norm, "rmsnorm", 2)
    check(
        xx["id"] == x["id"] and nk["id"] == nv["id"] == norm["id"],
        "projected cache shared RMS and original residual",
    )
    check(
        cc["id"] == c["id"] and ss["id"] == s["id"],
        "projected cache shared supplied pair coefficients",
    )
    sources = [x, gamma, wq, wk, wv, c, s, key, value, wo]
    check(
        all(n["op"] == "input" and not n["inputs"] for n in sources)
        and len({n["id"] for n in sources}) == len({n["host"] for n in sources}) == 10,
        "projected cache ten distinct supplied inputs",
    )
    out_by = {n["inputs"][0]: n for n in outs}
    ordered = sources + [
        norm,
        q,
        k,
        v,
        rq,
        rk,
        kt,
        score,
        prob,
        context,
        delta,
        add,
        out_by[add["id"]],
        out_by[rk["id"]],
        out_by[v["id"]],
    ]
    check(
        len({n["id"] for n in ordered}) == 25,
        "projected cache no hidden/dead/aliased stages",
    )
    m["nodes"] = ordered
    return m
