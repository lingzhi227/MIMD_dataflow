"""Shared typed RMS shape/precision checks, independent of physical layout."""

import copy, math
import numpy as np
from frontend import check


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    check(
        [n["op"] for n in m["nodes"]] == ["input", "input", "rmsnorm", "output"],
        "RMS typed two-input graph",
    )
    a, w, op, out = m["nodes"]
    check("dataflow" in op, "RMS requires an explicit dataflow policy")
    check(
        len({n["id"] for n in m["nodes"]}) == 4 and a["host"] != w["host"],
        "RMS unique ports and ids",
    )
    check(
        op["inputs"] == [a["id"], w["id"]] and out["inputs"] == [op["id"]],
        "RMS dependencies",
    )
    check(
        not m["states"] and not any("place" in n for n in m["nodes"]), "RMS owns region"
    )
    check(all(n.get("dtype") == "f16" for n in (a, w, op)), "RMS binary16 tensors")
    check(
        a["shape"] == op["shape"] and w["shape"] == [1, a["shape"][1]],
        "RMS row/features and weight shape",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 16,
        "RMS execution bounds",
    )
    check(
        math.isfinite(op["epsilon"]) and 0 < float(np.float16(op["epsilon"])) <= 1,
        "RMS finite representable positive half epsilon",
    )
    out.update(shape=op["shape"][:], dtype="f16")
    for n in m["nodes"]:
        n["interval"] = None
    m.update(epochs=epochs, input_bound=bound)
    return m
