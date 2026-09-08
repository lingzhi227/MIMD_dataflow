"""Attention plus output projection/residual; no cache mutation is implied."""

import copy
from frontend import check
from attention_ir import canonical as attention


def canonical(module):
    m = copy.deepcopy(module)
    ns = m["nodes"]
    by = {n["id"]: n for n in ns}
    check(
        len(by) == len(ns) == 12
        and sorted(n["op"] for n in ns)
        == sorted(
            ["input"] * 5 + ["matmul"] * 3 + ["transpose", "softmax", "add", "output"]
        ),
        "cache attention twelve unique nodes",
    )
    check(
        all(i in by for n in ns for i in n["inputs"]), "defined cache attention edges"
    )

    def operands(n, op, count):
        check(
            n["op"] == op and len(n["inputs"]) == count,
            "cache attention " + op + " edges",
        )
        return [by[i] for i in n["inputs"]]

    out = next(n for n in ns if n["op"] == "output")
    (add,) = operands(out, "output", 1)
    pair = operands(add, "add", 2)
    check(
        sorted(n["op"] for n in pair) == ["input", "matmul"],
        "cache residual and output projection",
    )
    x = next(n for n in pair if n["op"] == "input")
    delta = next(n for n in pair if n["op"] == "matmul")
    context, wo = operands(delta, "matmul", 2)
    check(wo["op"] == "input" and not wo["inputs"], "supplied output weight")
    subset = [
        n
        for n in ns
        if n["id"] not in {x["id"], wo["id"], delta["id"], add["id"], out["id"]}
    ]
    sub = attention(dict(m, nodes=subset + [dict(out, inputs=[context["id"]])]))
    q, k, v, t, score, sm, ctx, _ = sub["nodes"]
    sources = [x, q, k, v, wo]
    check(
        len({n["id"] for n in sources}) == len({n["host"] for n in sources}) == 5,
        "distinct cache attention inputs",
    )
    m["nodes"] = sources + [t, score, sm, ctx, delta, add, out]
    return m
