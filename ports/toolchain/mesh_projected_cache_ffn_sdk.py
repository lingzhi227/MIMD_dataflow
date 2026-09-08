"""SDK transport and output binding; no host numerical graph execution."""

import numpy as np
from projected_cache_ffn_codegen import extents
from projected_cache_ffn_transport import packed as pack
from mesh_cache_attention_sdk import unshard


def parameters(s):
    # This emitter specializes the verified layout parameters in CSL constants.
    return None


def packed(s, m, batch):
    return pack(s, batch)


def decode(s, m, raw):
    result = {}
    p = s["attention"]["P"]
    b = s["attention"]["B"]
    for host, v in s["output_bindings"].items():
        a = np.asarray(raw[v["physical"]], np.uint16).view(np.float16).astype(float)
        width = v["width"]
        start = v["offset"]
        a = a[:, :, start : start + b * (width // p)]
        result[host] = unshard(s["attention"], a, v["axis"], width).ravel().tolist()
    return result


def run(root):
    from half_region_runtime import run

    run(root, parameters, extents, packed, decode)


def audit(root):
    from integrity import verify_bundle, verify_codegen
    from half_region_runtime import read
    from mesh_projected_cache_ffn import plan
    from projected_cache_ffn_reference import audit_cases

    verify_bundle(root)
    verify_codegen(root)
    s, m, b, r = [
        read(root, n)
        for n in ("schedule.json", "semantic.json", "batches.json", "results.json")
    ]
    assert s == plan(m)
    return audit_cases(s, m, b, r)


def audit_cases(s, m, batches, r, *, require_complete=True):
    from projected_cache_ffn_reference import audit_cases as check

    return check(s, m, batches, r, require_complete=require_complete)
