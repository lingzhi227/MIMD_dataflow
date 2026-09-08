"""Ordinary f32 CG semantics for typed IR; independent validation lives in host."""

import math, struct
from frontend import check
from sparse_storage import CSC


def f(x):
    try:
        return struct.unpack("f", struct.pack("f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def product(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s = f(s + f(x * y))
    return s


def norm(a):
    if not all(math.isfinite(x) for x in a):
        return math.inf
    peak = max(map(abs, a), default=0)
    if peak == 0:
        return 0.0
    alpha = 2.0 ** max(-126, math.frexp(peak)[1] - 1)
    inv = f(1 / alpha)
    s = 0.0
    for x in a:
        v = f(x * inv)
        s = f(s + f(v * v))
    return f(f(math.sqrt(s)) * alpha)


def matrix_inputs(m, batch):
    ins = m["nodes"][:7]
    n = m["nodes"][7]["result_type"]["dimension"]
    check(set(batch) == {x["host"] for x in ins}, "CG input ports")
    for node in ins:
        values = batch[node["host"]]
        check(len(values) == node["shape"][0] * node["shape"][1], "CG input extent")
        if node["dtype"] == "u32":
            check(
                all(type(v) is int and 0 <= v <= 4294967295 for v in values),
                "CG u32 input",
            )
        else:
            check(
                all(
                    type(v) in (int, float)
                    and math.isfinite(v)
                    and abs(v) <= m["input_bound"]
                    and f(v) == v
                    for v in values
                ),
                "CG finite f32 input bound",
            )
    a = CSC(n, n, batch[ins[2]["host"]], batch[ins[1]["host"]], batch[ins[0]["host"]])
    limit = batch[ins[5]["host"]][0]
    tol = batch[ins[6]["host"]]
    check(
        limit <= m["nodes"][7]["result_type"]["max_iterations"]
        and all(v >= 0 for v in tol),
        "CG controls",
    )
    return a, list(batch[ins[3]["host"]]), list(batch[ins[4]["host"]]), limit, tol


def apply(a, x):
    y = [0.0] * a.rows
    for row, col, v in a.entries():
        y[row] = f(y[row] + f(v * x[col]))
    return y


def diagonal(a):
    out = [0.0] * a.rows
    for row, col, value in a.entries():
        if row == col:
            out[row] = value
    check(
        all(v >= 2**-16 for v in out),
        "Jacobi requires canonical diagonal entries at least2^-16",
    )
    return out


def solve(a, b, x, limit, tol, capacity, jacobi=False):
    inv = [f(1 / v) for v in diagonal(a)] if jacobi else None
    x = x.copy()
    r = [f(v - w) for v, w in zip(b, apply(a, x))]
    p = r.copy()
    rho = product(r, r)
    threshold = max(f(tol[0] * norm(b)), tol[1])
    history = [0.0] * (capacity + 1)
    history[0] = rho
    k = 0
    reason = 1
    if not math.isfinite(rho) or not math.isfinite(threshold):
        reason = 3
    elif norm(r) <= threshold:
        reason = 0
    elif rho <= 0:
        reason = 3
    else:
        for step in range(limit):
            if jacobi:
                z = [f(v * w) for v, w in zip(inv, r)]
                weighted = product(r, z)
                if not math.isfinite(weighted) or weighted <= 0:
                    reason = 3
                    break
                if step == 0:
                    p = z.copy()
                else:
                    beta = f(weighted / rho)
                    if not math.isfinite(beta):
                        reason = 3
                        break
                    p = [f(beta * v + w) for v, w in zip(p, z)]
                rho = weighted
            ap = apply(a, p)
            curvature = product(p, ap)
            if not math.isfinite(curvature):
                reason = 3
                break
            if curvature <= 0:
                reason = 2
                break
            alpha = f(rho / curvature)
            if not math.isfinite(alpha):
                reason = 3
                break
            x = [f(alpha * v + w) for v, w in zip(p, x)]
            r = [f(-alpha * v + w) for v, w in zip(ap, r)]
            nxt = product(r, r)
            k = step + 1
            history[k] = nxt
            if not math.isfinite(nxt):
                reason = 3
                break
            if nxt == 0 and norm(r) > threshold:
                reason = 3
                break
            if math.sqrt(nxt) <= threshold:
                reason = 0
                break
            if k == limit:
                break
            if not jacobi:
                beta = f(nxt / rho)
                if not math.isfinite(beta):
                    reason = 3
                    break
                p = [f(beta * v + w) for v, w in zip(p, r)]
                rho = nxt
    true = norm([f(v - w) for v, w in zip(b, apply(a, x))])
    if not math.isfinite(true):
        reason = 3
    elif reason == 0 and true > threshold:
        reason = 4
    return dict(
        solution=x,
        reason=[reason],
        iterations=[k],
        residual_squared=history,
        true_residual_norm=[true],
    )


def evaluate(m, batches):
    results = []
    check(len(batches) == m["epochs"], "CG epoch count")
    for batch in batches:
        a, b, x, limit, tol = matrix_inputs(m, batch)
        if m["nodes"][7]["op"] == "bicgstab_csc":
            from bicgstab_reference import solve as solve_bicgstab

            value = solve_bicgstab(
                a, b, x, limit, tol, m["nodes"][7]["result_type"]["max_iterations"]
            )
        else:
            value = solve(
                a,
                b,
                x,
                limit,
                tol,
                m["nodes"][7]["result_type"]["max_iterations"],
                jacobi=m["nodes"][7]["op"] == "pcg_csc",
            )
        results.append(
            {
                node["host"]: value[node["inputs"][0].split(".")[1]]
                for node in m["nodes"][8:]
            }
        )
    return results, {}
