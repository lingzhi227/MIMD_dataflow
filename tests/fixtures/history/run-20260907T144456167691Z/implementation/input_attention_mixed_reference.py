"""Native arithmetic interpretation; independent application oracle lives outside compiler."""

import numpy as np
from binary16 import matmul as half_matmul
from blocked_matmul import evaluate as blocked_matmul
from float32_arithmetic import matmul, softmax, rms
from mesh_pair_rotation import reference as rotation
from mesh_swiglu import standard
from mesh_input_attention import inputs


def stages(m, b):
    x, gamma, qw, kw, vw, c, s, o, u, g, d = inputs(m, b)
    nodes = m["nodes"]
    half = lambda a: np.asarray(a, np.float16).astype(float)
    norm = half(
        x * gamma / np.sqrt(np.mean(x * x, axis=1)[:, None] + nodes[11]["epsilon"])
    )
    qr, kr = half_matmul(norm, qw), half_matmul(norm, kw)
    v = matmul(norm, vw)
    q = half(rotation(qr, c, s, "odd_even", False)[1])
    k = half(rotation(kr, c, s, "odd_even", False)[1])
    score = half_matmul(q, k.T)
    probability = softmax(score, nodes[19]["scale"])
    attention = matmul(probability, v)
    projection = matmul(attention, o)
    z = np.float32(
        np.asarray(projection, np.float32) + np.asarray(x, np.float32)
    ).astype(float)
    normalized = half(rms(z, gamma, nodes[23]["epsilon"]))
    up = blocked_matmul(normalized, u, nodes[24]["block_size"])
    gate = blocked_matmul(normalized, g, nodes[25]["block_size"])
    activated = half(standard(up, gate)[0])
    hidden = half(up * activated)
    delta = blocked_matmul(hidden, d, nodes[28]["block_size"])
    result = half(np.float32(np.asarray(z, np.float32) + np.asarray(delta, np.float32)))
    return dict(
        input_normalized=norm,
        q_raw=qr,
        k_raw=kr,
        v_raw=v,
        q=q,
        k=k,
        v=v,
        score=score,
        probability=probability,
        attention=attention,
        projection=projection,
        z=z,
        normalized_z=normalized,
        delta=delta,
        result=result,
    )
