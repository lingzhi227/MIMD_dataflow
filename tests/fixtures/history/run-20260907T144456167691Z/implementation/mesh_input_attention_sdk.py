"""Transport declarations for the development31-node resident composition.

No host intermediate arithmetic. Final unified execution/audit dispatch remains
unregistered until precision and repeated-call evidence meet the full contract.
"""

import numpy as np
from mesh_input_attention import inputs
from mesh_attention_tail_sdk import extents as tail_extents
from mesh_mlp_sdk import packed as projection_packed
from mesh_common import pack_tiles


def extents(s):
    result = tail_extents(s)
    l, ow, p, nt = (s[k] for k in ("length", "output_weight_length", "P", "Nt"))
    sample = s["instrumentation"] == "sampled"
    result.update(
        q_weight=ow,
        k_weight=ow,
        v_weight=ow,
        cosine=nt // 2,
        sine=nt // 2,
        input_normalized=l,
        input_q_raw=l,
        input_k_raw=l,
        input_projection_history=3 * p * l if sample else 1,
        input_left_first=3 * l if sample else 1,
        input_right_first=3 * ow if sample else 1,
        input_pair_history=4 * l if sample else 1,
        input_prefix_progress=12,
    )
    return result


def packed(s, m, b):
    x, gamma, q, k, v, c, sine, o, u, g, d = inputs(m, b)
    p = s["P"]
    result = projection_packed(s, (x, u, g, d))
    result["residual"] = result.pop("x")
    for name, weight in (
        ("q_weight", q),
        ("k_weight", k),
        ("v_weight", v),
        ("output_weight", o),
    ):
        result[name] = projection_packed(s, (x, weight, weight, weight))["up_weight"]
    for name, array in (("gamma", gamma), ("cosine", c), ("sine", sine)):
        result[name] = np.broadcast_to(
            array.reshape(1, p, -1), (p, p, array.size // p)
        ).copy()
    return result
