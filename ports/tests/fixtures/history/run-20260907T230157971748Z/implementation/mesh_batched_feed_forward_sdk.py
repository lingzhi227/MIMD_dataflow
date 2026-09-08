"""Batch FFN half transport and all-PE stage inspection over SDK runtime."""

import numpy as np
from compiler_parameters import encode
from frontend import check
from input_contracts import validate_batch
from half_region_runtime import read
from mesh_batched_feed_forward import plan


def parameters(s):
    v = {k: s[k] for k in ("P", "B", "Nt", "Ft")}
    v.update(
        epsilon_bits=int(np.float16(s["epsilon"]).view(np.uint16)),
        sampled=int(s["instrumentation"] == "sampled"),
    )
    return encode({k: "u16" if k == "epsilon_bits" else "i16" for k in v}, v)


def extents(s):
    b, nt, ft = s["B"], s["Nt"], s["Ft"]
    sampled = s["instrumentation"] == "sampled"
    return dict(
        X=b * nt,
        gamma=nt,
        weights=3 * nt * ft,
        normalized=b * nt,
        scratch=b * nt,
        sums=s["padded_batches"],
        projections=2 * b * ft,
        activation=b * ft,
        hidden=b * ft,
        delta=b * nt,
        result=b * nt,
        history=2 * s["padded_batches"] if sampled else 1,
        partial=2 * b * ft if sampled else 1,
        down_partial=b * nt if sampled else 1,
        progress=8,
        timing=6,
        queues=2,
    )


def shard_y(s, a):
    return np.repeat(
        np.asarray(a)
        .reshape(s["B"], s["P"], s["Nt"])
        .transpose(1, 0, 2)
        .reshape(s["P"], 1, -1),
        s["P"],
        axis=1,
    )


def values(m, batch):
    validate_batch(m, batch)
    return [
        np.asarray(batch[n["host"]], float).reshape(n["shape"]) for n in m["nodes"][:5]
    ]


def packed(s, m, batch):
    x, gamma, wu, wg, wd = values(m, batch)
    p, nt, ft = s["P"], s["Nt"], s["Ft"]
    weights = np.zeros((p, p, 3 * nt * ft))
    for y in range(p):
        a = slice(y * nt, (y + 1) * nt)
        for col in range(p):
            b = slice(col * ft, (col + 1) * ft)
            weights[y, col] = np.concatenate(
                [wu[a, b].ravel(), wg[a, b].ravel(), wd[b, a].ravel()]
            )
    return dict(
        X=shard_y(s, x),
        gamma=np.repeat(gamma.reshape(p, 1, nt), p, axis=1),
        weights=weights,
    )


def decode(s, m, d):
    result = (
        np.asarray(d["result"], np.uint16)
        .view(np.float16)[:, 0]
        .reshape(s["P"], s["B"], s["Nt"])
        .transpose(1, 0, 2)
    )
    return {m["nodes"][-1]["host"]: result.astype(float).ravel().tolist()}


def run(root):
    from half_region_runtime import run as execute

    execute(root, parameters, extents, packed, decode)


def audit(root):
    from integrity import verify_bundle, verify_codegen
    from batched_ffn_reference import audit_cases

    verify_bundle(root)
    verify_codegen(root)
    s, m, b, r = [
        read(root, n)
        for n in ("schedule.json", "semantic.json", "batches.json", "results.json")
    ]
    check(s == plan(m), "batch FFN regenerated schedule")
    return audit_cases(s, m, b, r)


def audit_cases(s, m, batches, r, *, require_complete=True):
    from batched_ffn_reference import audit_cases as check_cases

    return check_cases(s, m, batches, r, require_complete=require_complete)
