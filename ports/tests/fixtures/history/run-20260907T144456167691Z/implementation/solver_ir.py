"""Typed resident solver contract. CSL scheduling is a separate implementation."""

import copy
from frontend import check


def result_type(n, capacity):
    check(
        type(n) is int and n > 0 and type(capacity) is int and 1 <= capacity <= 256,
        "solver result dimensions",
    )
    return dict(
        kind="record",
        name="solver_result",
        dimension=n,
        max_iterations=capacity,
        fields={
            "solution": dict(dtype="f32", shape=[n, 1]),
            "reason": dict(dtype="u32", shape=[1, 1]),
            "iterations": dict(dtype="u32", shape=[1, 1]),
            "residual_squared": dict(dtype="f32", shape=[capacity + 1, 1]),
            "true_residual_norm": dict(dtype="f32", shape=[1, 1]),
        },
    )


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    nodes = m["nodes"]
    check(
        len(nodes) == 13,
        "solver graph requires seven inputs, one solve and five result outputs",
    )
    ins, solve, outs = nodes[:7], nodes[7], nodes[8:]
    check(
        [x["op"] for x in ins]
        == [
            "input",
            "index_input",
            "index_input",
            "input",
            "input",
            "index_input",
            "input",
        ]
        and solve["op"] in ("cg_csc", "pcg_csc", "bicgstab_csc")
        and all(x["op"] == "output" for x in outs),
        "solver graph operations",
    )
    t = solve["result_type"]
    n, cap = t["dimension"], t["max_iterations"]
    check(t == result_type(n, cap), "solver record schema")
    nnz = ins[0]["shape"][0]
    check(
        [x["shape"] for x in ins]
        == [[nnz, 1], [nnz, 1], [n + 1, 1], [n, 1], [n, 1], [1, 1], [2, 1]],
        "solver input shapes",
    )
    check(
        1 <= n <= 8192 and 1 <= nnz <= min(n * n, 262144), "solver sparse shape bounds"
    )
    check(solve["inputs"] == [x["id"] for x in ins], "solver dependencies")
    check(len({x["id"] for x in nodes}) == len(nodes), "solver unique IDs")
    check(
        len({x["host"] for x in ins}) == 7 and len({x["host"] for x in outs}) == 5,
        "solver unique ports",
    )
    check(
        {tuple(x["inputs"]) for x in outs}
        == {(solve["id"] + "." + field,) for field in t["fields"]},
        "solver requires each result field exactly once",
    )
    d = solve["dataflow"]
    expected = dict(
        storage="csc",
        exchange="trains",
        reduce="row_column",
        redistribute="transpose",
        recurrence="resident",
        compute="vector",
        fp="relaxed",
    )
    check(
        set(d)
        == set(expected) | {"rows", "cols", "nnz_per_pe", "cols_per_pe", "rows_per_pe"},
        "solver dataflow keys",
    )
    check(all(d[k] == v for k, v in expected.items()), "solver dataflow policy")
    check(
        type(d["rows"]) is int
        and d["rows"] in (4, 8)
        and d["cols"] == d["rows"]
        and n % (d["rows"] * d["cols"]) == 0,
        "solver square mesh and divisible vectors",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 32767,
        "solver execution bounds",
    )
    check(
        not m["states"]
        and not any("place" in x for x in nodes)
        and not any("dataflow" in x for x in nodes if x is not solve),
        "solver owns resident layout",
    )
    for x in ins:
        x["dtype"] = "u32" if x["op"] == "index_input" else "f32"
    for x in outs:
        field = x["inputs"][0].split(".")[1]
        x.update(t["fields"][field])
    solve["shape"] = None
    m.update(profile="mesh_cg.v1", epochs=epochs, input_bound=bound)
    return m
