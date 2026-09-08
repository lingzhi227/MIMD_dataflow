"""Typed fixed-step power result and reference semantics; distinct from solver convergence."""

import copy, math
from frontend import check
from sparse_storage import CSC
from solver_reference import f, norm, apply


def result_type(n, capacity):
    check(
        type(n) is int
        and 1 <= n <= 8192
        and type(capacity) is int
        and 1 <= capacity <= 32,
        "power record dimensions",
    )
    return dict(
        kind="record",
        name="power_result",
        dimension=n,
        max_iterations=capacity,
        fields=dict(
            vector=dict(dtype="f32", shape=[n, 1]),
            reason=dict(dtype="u32", shape=[1, 1]),
            iterations=dict(dtype="u32", shape=[1, 1]),
            norms=dict(dtype="f32", shape=[capacity, 1]),
        ),
    )


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    nodes = m["nodes"]
    check(len(nodes) == 10, "power requires five inputs and four result outputs")
    ins, op, outs = nodes[:5], nodes[5], nodes[6:]
    t = op["result_type"]
    n = t["dimension"]
    cap = t["max_iterations"]
    nnz = ins[0]["shape"][0]
    check(t == result_type(n, cap), "power result type")
    check(
        [x["op"] for x in ins]
        == ["input", "index_input", "index_input", "input", "index_input"]
        and op["op"] == "power_csc"
        and all(x["op"] == "output" for x in outs),
        "power operations",
    )
    check(
        [x["shape"] for x in ins] == [[nnz, 1], [nnz, 1], [n + 1, 1], [n, 1], [1, 1]]
        and 1 <= nnz <= min(n * n, 262144),
        "power input shape",
    )
    check(
        op["inputs"] == [x["id"] for x in ins]
        and len({x["id"] for x in nodes}) == len(nodes),
        "power dependencies",
    )
    check(
        len({x["host"] for x in ins}) == 5 and len({x["host"] for x in outs}) == 4,
        "power ports",
    )
    check(
        {tuple(x["inputs"]) for x in outs}
        == {(op["id"] + "." + field,) for field in t["fields"]},
        "power result fields",
    )
    d = op["dataflow"]
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
        == set(expected) | {"rows", "cols", "nnz_per_pe", "cols_per_pe", "rows_per_pe"}
        and all(d[k] == v for k, v in expected.items()),
        "power dataflow policy",
    )
    check(
        type(d["rows"]) is int
        and d["rows"] in (4, 8)
        and d["cols"] == d["rows"]
        and n % (d["rows"] * d["cols"]) == 0,
        "power square mesh",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 32767,
        "power execution bounds",
    )
    check(
        not m["states"]
        and not any("place" in x for x in nodes)
        and not any("dataflow" in x for x in nodes if x is not op),
        "power owns resident schedule",
    )
    for x in ins:
        x["dtype"] = "u32" if x["op"] == "index_input" else "f32"
    for x in outs:
        x.update(t["fields"][x["inputs"][0].split(".")[1]])
    op["shape"] = None
    m.update(profile="mesh_power.v1", epochs=epochs, input_bound=bound)
    return m


def inputs(m, b):
    ins = m["nodes"][:5]
    check(set(b) == {n["host"] for n in ins}, "power input ports")
    for n in ins:
        v = b[n["host"]]
        check(len(v) == math.prod(n["shape"]), "power input extent")
        if n["dtype"] == "u32":
            check(
                all(type(x) is int and 0 <= x < 2**32 for x in v),
                "power integer transport",
            )
        else:
            check(
                all(
                    type(x) in (int, float)
                    and math.isfinite(x)
                    and abs(x) <= m["input_bound"]
                    and x == f(x)
                    for x in v
                ),
                "power finite f32 inputs",
            )
    n = m["nodes"][5]["result_type"]["dimension"]
    steps = b[ins[4]["host"]][0]
    check(
        steps <= m["nodes"][5]["result_type"]["max_iterations"], "power runtime budget"
    )
    return (
        CSC(n, n, b[ins[2]["host"]], b[ins[1]["host"]], b[ins[0]["host"]]),
        list(b[ins[3]["host"]]),
        steps,
    )


def solve(a, x, steps, capacity):
    x = x.copy()
    norms = [0.0] * capacity
    k = 0
    reason = 0
    for i in range(steps):
        y = apply(a, x)
        nr = norm(y)
        norms[i] = nr
        if not math.isfinite(nr):
            reason = 2
            break
        if nr == 0:
            reason = 1
            break
        inverse = f(1.0 / nr)
        if not math.isfinite(inverse):
            reason = 2
            break
        x = [f(v * inverse) for v in y]
        k = i + 1
    return dict(vector=x, reason=[reason], iterations=[k], norms=norms)


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "power epochs")
    results = []
    for b in batches:
        a, x, steps = inputs(m, b)
        value = solve(a, x, steps, m["nodes"][5]["result_type"]["max_iterations"])
        results.append(
            {n["host"]: value[n["inputs"][0].split(".")[1]] for n in m["nodes"][6:]}
        )
    return results, {}
