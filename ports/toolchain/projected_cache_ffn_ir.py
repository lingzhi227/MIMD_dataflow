"""Structural composition of the existing attention and FFN graph contracts."""

import copy
from frontend import check
from projected_cache_ir import canonical as attention_canonical
from feed_forward_ir import canonical as ffn_canonical


def canonical(module):
    m = copy.deepcopy(module)
    by = {n["id"]: n for n in m["nodes"]}
    check(len(by) == len(m["nodes"]) == 35, "attention FFN thirty-five unique nodes")
    check(
        all(i in by for n in by.values() for i in n["inputs"]),
        "attention FFN defined edges",
    )
    outs = [n for n in by.values() if n["op"] == "output"]
    finals = [
        n for n in outs if len(n["inputs"]) == 1 and by[n["inputs"][0]]["op"] == "add"
    ]
    check(len(finals) == 1 and len(outs) == 3, "attention FFN three semantic outputs")
    out = finals[0]
    final = by[out["inputs"][0]]

    def args(n, op, count):
        check(
            n["op"] == op and len(n["inputs"]) == count,
            "attention FFN " + op + " edges",
        )
        return [by[i] for i in n["inputs"]]

    residual = args(final, "add", 2)
    z = [n for n in residual if n["op"] == "add"]
    down = [n for n in residual if n["op"] == "matmul"]
    check(len(z) == len(down) == 1, "FFN residual is attention result Z")
    z, down = z[0], down[0]
    hidden, wd = args(down, "matmul", 2)
    factors = args(hidden, "multiply", 2)
    ups = [n for n in factors if n["op"] == "matmul"]
    acts = [n for n in factors if n["op"] == "silu"]
    check(len(ups) == len(acts) == 1, "FFN UP times SiLU GATE")
    up, act = ups[0], acts[0]
    gate = args(act, "silu", 1)[0]
    norm, wu = args(up, "matmul", 2)
    gnorm, wg = args(gate, "matmul", 2)
    source, gamma = args(norm, "rmsnorm", 2)
    check(
        gnorm["id"] == norm["id"] and source["id"] == z["id"], "FFN shared normalized Z"
    )
    # Reuse the exact original attention graph verifier with only its output edge
    # restored to the internal Z boundary. The actual graph remains unchanged.
    prefix_out = copy.deepcopy(out)
    prefix_out["inputs"] = [z["id"]]
    prefix_outputs = [prefix_out if n["id"] == out["id"] else n for n in outs]
    live = {}
    visiting = set()

    def visit(n):
        check(n["id"] not in visiting, "attention FFN acyclic prefix")
        if n["id"] in live:
            return
        visiting.add(n["id"])
        for i in n["inputs"]:
            visit(by[i])
        visiting.remove(n["id"])
        live[n["id"]] = n

    for n in prefix_outputs:
        for i in n["inputs"]:
            visit(by[i])
    prefix = attention_canonical(dict(m, nodes=list(live.values()) + prefix_outputs))
    check(
        prefix["nodes"][1]["id"] == gamma["id"],
        "Decode source-shared gamma across RMS stages",
    )
    stub = copy.deepcopy(z)
    stub.update(op="input", inputs=[], host="__derived_attention_z", abs_bound=0.0)
    tail_nodes = [
        stub,
        gamma,
        wu,
        wg,
        wd,
        norm,
        up,
        gate,
        act,
        hidden,
        down,
        final,
        out,
    ]
    tail = ffn_canonical(dict(m, nodes=tail_nodes))
    covered = {n["id"] for n in prefix["nodes"]} | {n["id"] for n in tail["nodes"]}
    check(covered == set(by), "attention FFN has no detached nodes")
    return dict(
        module=m,
        attention=prefix,
        ffn=tail,
        boundary=dict(
            producer=z["id"],
            consumer=norm["id"],
            gamma=gamma["id"],
            virtual_input_requires_parent_range=True,
        ),
        status="structural composition only; parent ranges, placement and code generation required",
    )
