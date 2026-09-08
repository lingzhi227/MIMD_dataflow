"""Typed matrix input contracts and native-order oracle, independent of topology."""

from frontend import check


def inputs(m, b):
    import numpy as np

    nodes = m["nodes"][:2]
    check(set(b) == {n["host"] for n in nodes}, "half input ports")
    arrays = []
    for n in nodes:
        v = b[n["host"]]
        check(len(v) == n["shape"][0] * n["shape"][1], "half input extent")
        check(all(type(x) in (int, float) for x in v), "half scalar input types")
        a = np.asarray(v, dtype=float).reshape(n["shape"])
        check(
            np.all(np.isfinite(a)) and np.all(np.abs(a) <= m["input_bound"]),
            "half input bounds",
        )
        check(
            np.array_equal(a, a.astype(np.float16).astype(float)),
            "half exactly representable inputs",
        )
        arrays.append(a)
    return arrays


def evaluate(m, batches):
    from binary16 import matmul

    check(len(batches) == m["epochs"], "half epoch count")
    return [
        {m["nodes"][3]["host"]: matmul(*inputs(m, b)).ravel().tolist()} for b in batches
    ], {}
