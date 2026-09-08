"""Structured-grid semantics: seven-sample neighborhoods, resident pencil iteration."""

import copy, math
from frontend import check
from float32 import f32
from local_kernel import evaluate_kernel


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    ns = m["nodes"]
    check(
        [n["op"] for n in ns] == ["input", "input", "grid_iterate", "output"],
        "grid profile requires field, coefficients, iteration, output",
    )
    field, coeff, op, out = ns
    g = op["grid"]
    x, y, z, steps = (g[k] for k in ("x", "y", "z", "steps"))
    check(
        all(type(v) is int and v > 0 for v in (x, y, z, steps))
        and x <= 16
        and y <= 16
        and z <= 256
        and steps <= 128,
        "grid bounds: x/y <=16, z <=256, steps <=128",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 32767,
        "execution bounds",
    )
    check(
        op["inputs"] == [field["id"], coeff["id"]] and out["inputs"] == [op["id"]],
        "grid dependencies",
    )
    check(
        field["shape"] == op["shape"] == [x * y, z]
        and coeff["shape"][0] == 1
        and coeff["shape"][1] <= 256,
        "grid shapes",
    )
    check(
        op["body"]["arrays"]["a"] == [7, z]
        and op["body"]["arrays"]["b"] == coeff["shape"],
        "neighborhood/coefficients lambda shapes",
    )
    check(
        not m["states"] and not any("place" in n for n in ns),
        "grid owns placement and state; scalar graph placement is unsupported",
    )
    check(
        len({n["id"] for n in ns}) == 4 and field["host"] != coeff["host"],
        "unique ids and input ports",
    )
    out["shape"] = op["shape"]
    m.update(profile="grid.v1", epochs=epochs, input_bound=bound)
    return m


def simulate(m, batch, trace=False):
    field, coeff, op, out = m["nodes"]
    g = op["grid"]
    nx, ny, z, steps = (g[k] for k in ("x", "y", "z", "steps"))
    check(set(batch) == {field["host"], coeff["host"]}, "grid input names")
    for n in (field, coeff):
        check(
            len(batch[n["host"]]) == math.prod(n["shape"])
            and all(
                type(v) in (int, float)
                and math.isfinite(v)
                and abs(v) <= m["input_bound"]
                for v in batch[n["host"]]
            ),
            "grid input shape/bounds",
        )
    values = [f32(v) for v in batch[field["host"]]]
    weights = [f32(v) for v in batch[coeff["host"]]]
    history = {f"p{x}_{y}": [] for x in range(nx) for y in range(ny)}
    for step in range(steps):
        nxt = [0.0] * len(values)
        for x in range(nx):
            for y in range(ny):
                at = (x * ny + y) * z
                center = values[at : at + z]
                halo = []
                for xx, yy in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    off = (xx * ny + yy) * z
                    halo.extend(
                        values[off : off + z]
                        if 0 <= xx < nx and 0 <= yy < ny
                        else [0.0] * z
                    )
                halo.extend([0.0] + center[:-1])
                halo.extend(center[1:] + [0.0])
                halo.extend(center)
                v = evaluate_kernel(op["body"], [halo, weights])
                nxt[at : at + z] = v
                if trace:
                    history[f"p{x}_{y}"].extend(v)
        values = nxt
    return values, history


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "epoch count")
    field, coeff, op, out = m["nodes"]
    history = {n["id"]: [] for n in m["nodes"]}
    outputs = []
    for batch in batches:
        values, _ = simulate(m, batch)
        for n in (field, coeff):
            history[n["id"]].extend(batch[n["host"]])
        history[op["id"]].extend(values)
        history[out["id"]].extend(values)
        outputs.append({out["host"]: values})
    return outputs, history
