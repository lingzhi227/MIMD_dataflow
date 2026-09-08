"""Original-input normalized FFN mathematics; never imported by compiler/codegen."""

import math
import numpy as np


def batches(m, n, f, p):
    from mlp_fixtures import batches as mlp_batches

    raw = mlp_batches(m, n, f, p)
    rng = np.random.default_rng(210113)
    out = []
    for epoch, b in enumerate(raw):
        item = {
            "z": b["x"],
            "gamma": rng.uniform(0.5, 1.5, (1, n))
            .astype(np.float16)
            .astype(float)
            .ravel()
            .tolist(),
        }
        for key in ("up_weight", "gate_weight", "down_weight"):
            item[key] = (
                (np.asarray(b[key]) / 16).astype(np.float16).astype(float).tolist()
            )
        if epoch == 5:
            item["gamma"] = [1.5] * n
        out.append(item)
    return out


def reference(m, n, f, epsilon, b):
    z = np.asarray(b["z"]).reshape(m, n)
    gamma = np.asarray(b["gamma"]).reshape(1, n)
    u = np.asarray(b["up_weight"]).reshape(n, f)
    g = np.asarray(b["gate_weight"]).reshape(n, f)
    d = np.asarray(b["down_weight"]).reshape(f, n)

    def product(a, b):
        return np.asarray(
            [
                [
                    math.fsum(float(x) * float(y) for x, y in zip(row, col))
                    for col in b.T
                ]
                for row in a
            ]
        )

    means = np.asarray([math.fsum(float(v) * float(v) for v in row) / n for row in z])
    normalized = z * gamma / np.sqrt(means[:, None] + epsilon)
    up = product(normalized, u)
    gate = product(normalized, g)
    delta = product(up * gate / (1 + np.exp(-gate)), d)
    return z + delta, delta


def check(m, n, f, epsilon, b, out, delta=None):
    expected, expected_delta = reference(m, n, f, epsilon, b)
    assert set(out) == {"output"}

    def measure(actual, target):
        actual = np.asarray(actual).reshape(m, n)
        error = actual - target
        l2 = float(np.linalg.norm(error)) / max(float(np.linalg.norm(target)), 1e-30)
        peak = float(np.max(np.abs(error))) / max(float(np.max(np.abs(target))), 1e-30)
        assert np.all(np.isfinite(actual)) and l2 <= 0.02 and peak <= 0.03, (l2, peak)
        return dict(relative_l2=l2, peak_scaled_error=peak, fixed_accuracy_passed=True)

    result = dict(
        measure(out["output"], expected),
        contract="normalized-feed-forward-half-normwise-v1",
    )
    if delta is not None:
        result["mlp_delta"] = measure(delta, expected_delta)
    return result
