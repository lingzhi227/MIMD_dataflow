"""Original-input mathematics for the pending shared-gamma input/attention chain."""



import math
import numpy as np
from attention_tail_fixtures import (
    batches as tail_batches,
    reference as tail_reference,
    check as tail_check,
)


def batches(m=64, n=64, f=256, p=8):
    rng = np.random.default_rng(210131)
    out = []
    for epoch, old in enumerate(tail_batches(m, n, f, p)):
        b = {k: v for k, v in old.items() if k not in ("q", "k", "v", "residual")}
        b["input_x"] = old["residual"]
        for key in ("q", "k", "v"):
            a = rng.uniform(-0.00390625, 0.00390625, (n, n)).astype(np.float16)
            if epoch in (2, 7) and key == "v":
                a[:] = 0
            if epoch == 6:
                a[:] = 0
            if epoch == 3:
                a[:] = -0.00390625 if key == "k" else 0.00390625
            b[key + "_weight"] = a.astype(float).ravel().tolist()
        angle = rng.uniform(-math.pi, math.pi, (1, n // 2))
        if epoch in (0, 3, 6):
            angle[:] = 0
        if epoch == 5:
            angle[:] = math.pi / 2
        b["cosine"] = np.cos(angle).astype(np.float16).astype(float).ravel().tolist()
        b["sine"] = np.sin(angle).astype(np.float16).astype(float).ravel().tolist()
        out.append(b)
    return out


def prefix_reference(m, n, epsilon, b):
    x = np.asarray(b["input_x"]).reshape(m, n)
    gamma = np.asarray(b["gamma"]).reshape(1, n)
    inverse = np.array(
        [
            1 / math.sqrt(math.fsum(float(v) * float(v) for v in row) / n + epsilon)
            for row in x
        ]
    )
    normalized = x * gamma * inverse[:, None]
    projections = {}
    for name in ("q", "k", "v"):
        w = np.asarray(b[name + "_weight"]).reshape(n, n)
        projections[name + "_raw"] = np.array(
            [
                [
                    math.fsum(float(a) * float(c) for a, c in zip(row, column))
                    for column in w.T
                ]
                for row in normalized
            ]
        )
    cosine = np.asarray(b["cosine"]).reshape(1, n // 2)
    sine = np.asarray(b["sine"]).reshape(1, n // 2)

    def rotate(x):
        r = np.empty_like(x)
        r[:, ::2] = x[:, 1::2] * cosine - x[:, ::2] * sine
        r[:, 1::2] = x[:, ::2] * cosine + x[:, 1::2] * sine
        return r

    return dict(
        input_normalized=normalized,
        **projections,
        q=rotate(projections["q_raw"]),
        k=rotate(projections["k_raw"]),
        v=projections["v_raw"]
    )


def child_inputs(m, n, epsilon, b):
    prefix = prefix_reference(m, n, epsilon, b)
    child = {
        key: b[key]
        for key in ("gamma", "output_weight", "up_weight", "gate_weight", "down_weight")
    }
    child["residual"] = b["input_x"]
    child.update({key: prefix[key].ravel().tolist() for key in ("q", "k", "v")})
    return prefix, child


def check(m, n, f, epsilon, scale, b, out, observations):
    prefix, child = child_inputs(m, n, epsilon, b)
    result = tail_check(
        m,
        n,
        f,
        epsilon,
        scale,
        child,
        out,
        observations["attention"],
        observations["projection"],
        observations["delta"],
        score=observations["score"],
        probability=observations["probability"],
    )
    result["contract"] = "shared-gamma-input-attention-tail-half-normwise-v1"
    for name, expected in prefix.items():
        value = np.asarray(observations[name]).reshape(m, n)
        error = value - expected
        l2 = float(np.linalg.norm(error)) / max(float(np.linalg.norm(expected)), 1e-30)
        peak = float(np.max(np.abs(error))) / max(
            float(np.max(np.abs(expected))), 1e-30
        )
        limits = (0.01, 0.015) if name == "input_normalized" else (0.02, 0.03)
        assert np.all(np.isfinite(value)) and l2 <= limits[0] and peak <= limits[1], (
            name,
            l2,
            peak,
        )
        result[name] = dict(
            relative_l2=l2, peak_scaled_error=peak, fixed_accuracy_passed=True
        )
    # This local transform check is separate from the original-input whole-chain
    # oracle above: projected input rounding errors have already been measured.
    c = np.asarray(b["cosine"]).reshape(1, n // 2)
    s = np.asarray(b["sine"]).reshape(1, n // 2)
    for name in ("q", "k"):
        raw = np.asarray(observations[name + "_raw"]).reshape(m, n)
        a, bb = raw[:, 1::2], raw[:, ::2]
        wanted = np.empty_like(raw)
        magnitude = np.empty_like(raw)
        wanted[:, ::2] = a * c - bb * s
        wanted[:, 1::2] = bb * c + a * s
        magnitude[:, ::2] = np.abs(a * c) + np.abs(bb * s)
        magnitude[:, 1::2] = np.abs(bb * c) + np.abs(a * s)
        error = np.abs(np.asarray(observations[name]).reshape(m, n) - wanted)
        assert np.all(error <= 0.0015 * magnitude + 2**-23), (
            name,
            "local pair rounding",
        )
        result[name]["local_pair_rounding_passed"] = True
    return result
