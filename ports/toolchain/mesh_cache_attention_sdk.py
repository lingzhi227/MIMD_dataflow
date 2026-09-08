"""Physical cache serialization and raw SDK transport; no host contractions."""

import numpy as np
from compiler_parameters import encode
from frontend import check
from half_region_runtime import read
from mesh_cache_attention import plan, values


def parameters(s):
    v = {
        k: s[k]
        for k in ("P", "B", "Nt", "St", "score_block", "value_block", "output_block")
    }
    v.update(
        scale_bits=int(np.float16(s["scale"]).view(np.uint16)),
        sampled=int(s["instrumentation"] == "sampled"),
    )
    return encode({k: "u16" if k == "scale_bits" else "i16" for k in v}, v)


def extents(s):
    return {
        **{
            k: v // 2
            for k, v in s["numeric_allocations"].items()
            if not k.startswith(("collective_", "max_")) and "_block_" not in k
        },
        "progress": 8,
        "timing": 6,
        "queues": 2,
    }


def shard(s, a, axis):
    a = np.asarray(a)
    b, width = a.shape
    p = s["P"]
    local = width // p
    chunks = a.reshape(b, p, local).transpose(1, 0, 2).reshape(p, b * local)
    return (
        np.repeat(chunks[:, None], p, axis=1)
        if axis == "y"
        else np.repeat(chunks[None], p, axis=0)
    )


def unshard(s, raw, axis, width):
    a = np.asarray(raw)
    tiles = a[:, 0] if axis == "y" else a[0]
    return (
        tiles.reshape(s["P"], s["B"], width // s["P"])
        .transpose(1, 0, 2)
        .reshape(s["B"], width)
    )


def packed(s, m, batch):
    x, q, k, v, w = values(m, batch)
    p, nt, st = s["P"], s["Nt"], s["St"]
    keys = np.zeros((p, p, nt * st))
    vals = keys.copy()
    weights = np.zeros((p, p, nt * nt))
    for y in range(p):
        seq = slice(y * st, (y + 1) * st)
        output = slice(y * nt, (y + 1) * nt)
        for col in range(p):
            feat = slice(col * nt, (col + 1) * nt)
            keys[y, col] = k[seq, feat].T.ravel()
            vals[y, col] = v[seq, feat].ravel()
            weights[y, col] = w[feat, output].ravel()
    return dict(X=shard(s, x, "y"), Q=shard(s, q, "x"), K=keys, V=vals, W=weights)


def decode(s, m, d):
    a = np.asarray(d["result"], np.uint16).view(np.float16).astype(float)
    return {m["nodes"][-1]["host"]: unshard(s, a, "y", s["N"]).ravel().tolist()}


def run(root):
    from half_region_runtime import run as execute

    execute(root, parameters, extents, packed, decode)


def audit(root):
    from integrity import verify_bundle, verify_codegen

    verify_bundle(root)
    verify_codegen(root)
    s, m, b, r = [
        read(root, n)
        for n in ("schedule.json", "semantic.json", "batches.json", "results.json")
    ]
    check(s == plan(m), "regenerated cache attention schedule")
    return audit_cases(s, m, b, r)


def audit_cases(s, m, batches, r, *, require_complete=True):
    from cache_attention_reference import audit_cases as inspect

    return inspect(s, m, batches, r, require_complete=require_complete)
