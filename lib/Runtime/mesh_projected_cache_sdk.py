"""Initial physical inputs and final/intermediate readback; no host graph arithmetic."""

import numpy as np
from frontend import check
from compiler_parameters import encode
from half_region_runtime import read
from mesh_projected_cache import plan, values
from mesh_cache_attention_sdk import shard, unshard


def parameters(s):
    v = {
        k: s[k]
        for k in ("P", "B", "Nt", "St", "score_block", "value_block", "output_block")
    }
    v.update(dict(zip(("q_block", "k_block", "v_block"), s["projection_blocks"])))
    v.update(
        scale_bits=int(np.float16(s["scale"]).view(np.uint16)),
        epsilon_bits=int(np.float16(s["epsilon"]).view(np.uint16)),
        sampled=int(s["instrumentation"] == "sampled"),
    )
    return encode({k: "u16" if k.endswith("_bits") else "i16" for k in v}, v)


def extents(s):
    return {
        **{
            k: v // 2
            for k, v in s["numeric_allocations"].items()
            if not k.startswith(("collective_", "max_")) and "_block_" not in k
        },
        "progress": 11,
        "timing": 6,
        "queues": 2,
    }


def packed(s, m, batch):
    x, gamma, wq, wk, wv, c, sn, k, v, wo = values(m, batch)
    p, nt, st = s["P"], s["Nt"], s["St"]

    def tiles(a, feature_major=False):
        out = np.empty((p, p, nt * st))
        for y in range(p):
            for col in range(p):
                tile = a[y * st : (y + 1) * st, col * nt : (col + 1) * nt]
                out[y, col] = (tile.T if feature_major else tile).ravel()
        return out

    weights = np.empty((p, p, 3 * nt * nt))
    output = np.empty((p, p, nt * nt))
    for y in range(p):
        for col in range(p):
            weights[y, col] = np.concatenate(
                [
                    w[y * nt : (y + 1) * nt, col * nt : (col + 1) * nt].ravel()
                    for w in (wq, wk, wv)
                ]
            )
            output[y, col] = wo[
                col * nt : (col + 1) * nt, y * nt : (y + 1) * nt
            ].ravel()
    return dict(
        X=shard(s, x, "y"),
        gamma=shard(s, gamma, "y"),
        qkv_weights=weights,
        cosine=shard(s, c, "x"),
        sine=shard(s, sn, "x"),
        K=tiles(k, True),
        V=tiles(v),
        W=output,
    )


def decode(s, m, d):
    half = lambda k: np.asarray(d[k], np.uint16).view(np.float16).astype(float)
    value = half("projections")[:, :, 2 * s["B"] * s["Nt"] :]
    return {
        m["nodes"][22]["host"]: unshard(s, half("result"), "y", s["N"])
        .ravel()
        .tolist(),
        m["nodes"][23]["host"]: unshard(s, half("rotated_key"), "x", s["N"])
        .ravel()
        .tolist(),
        m["nodes"][24]["host"]: unshard(s, value, "x", s["N"]).ravel().tolist(),
    }


def run(root):
    from half_region_runtime import run as execute

    execute(root, parameters, extents, packed, decode)


def audit(root):
    from integrity import verify_bundle, verify_codegen

    verify_bundle(root)
    verify_codegen(root)
    s, m, bs, r = [
        read(root, n)
        for n in ("schedule.json", "semantic.json", "batches.json", "results.json")
    ]
    check(s == plan(m), "projected cache schedule regeneration")
    return audit_cases(s, m, bs, r)


def audit_cases(s, m, bs, r, *, require_complete=True):
    from projected_cache_reference import audit_cases as inspect

    return inspect(s, m, bs, r, require_complete=require_complete)
