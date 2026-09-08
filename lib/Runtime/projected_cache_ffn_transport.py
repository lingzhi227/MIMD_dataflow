"""Pure checked physical packing for the composed resident graph."""

import numpy as np
from frontend import check
from mesh_projected_cache import values
from mesh_projected_cache_sdk import packed as prefix_packed


def prefix_batch(s, batch):
    return {n["host"]: batch[n["host"]] for n in s["graph"]["attention"]["nodes"][:10]}


def inputs(s, batch):
    from input_contracts import validate_batch, effective_bound

    module = s["graph"]["module"]
    declared = [n for n in module["nodes"] if n["op"] == "input"]
    check(set(batch) == {n["host"] for n in declared}, "composed thirteen input ports")
    validate_batch(module, batch)
    for n in declared:
        data = np.asarray(batch[n["host"]], float)
        check(
            data.size == np.prod(n["shape"])
            and np.all(np.isfinite(data))
            and np.all(np.abs(data) <= effective_bound(n, module["input_bound"]))
            and np.array_equal(data, data.astype(np.float16).astype(float)),
            "composed exact bounded half inputs",
        )
    ns = s["graph"]["ffn"]["nodes"]
    weights = [np.asarray(batch[n["host"]], float).reshape(n["shape"]) for n in ns[2:5]]
    return values(s["graph"]["attention"], prefix_batch(s, batch)), weights


def packed(s, batch):
    a = s["attention"]
    p, nt, ft = a["P"], a["Nt"], s["Ft"]
    result = prefix_packed(a, s["graph"]["attention"], prefix_batch(s, batch))
    _, weights = inputs(s, batch)
    wu, wg, wd = weights
    out = np.empty((p, p, 3 * nt * ft))
    for y in range(p):
        for x in range(p):
            out[y, x] = np.concatenate(
                [
                    wu[y * nt : (y + 1) * nt, x * ft : (x + 1) * ft].ravel(),
                    wg[y * nt : (y + 1) * nt, x * ft : (x + 1) * ft].ravel(),
                    wd[x * ft : (x + 1) * ft, y * nt : (y + 1) * nt].ravel(),
                ]
            )
    result["ffn_weights"] = out
    return result
