"""Original-input probability/value and general matrix fixtures, independent math.fsum."""



import math
import numpy as np


def batches(m, n):
    rng = np.random.default_rng(210109)
    prob = np.asarray(rng.uniform(0, 1, (m, m)), np.float16).astype(float)
    prob = np.asarray(prob / prob.sum(axis=1)[:, None], np.float16).astype(float)
    value = np.asarray(rng.uniform(-0.25, 0.25, (m, n)), np.float16).astype(float)
    pairs = [
        (np.eye(m), value),
        (prob, value),
        (np.zeros((m, m)), rng.uniform(-0.25, 0.25, (m, n))),
        (np.full((m, m), 1 / m), value),
        (np.roll(prob, 1, axis=1), rng.uniform(-0.25, 0.25, (m, n))),
        (rng.uniform(-0.125, 0.125, (m, m)), rng.uniform(-0.25, 0.25, (m, n))),
    ]
    return [
        dict(
            a=np.asarray(a, np.float16).astype(float).ravel().tolist(),
            b=np.asarray(b, np.float16).astype(float).ravel().tolist(),
        )
        for a, b in pairs
    ]


def check(m, n, b, o):
    assert set(b) == {"a", "b"} and set(o) == {"product"}
    left = np.asarray(b["a"]).reshape(m, m)
    right = np.asarray(b["b"]).reshape(m, n)
    actual = np.asarray(o["product"]).reshape(m, n)
    nominal = np.array(
        [
            [
                math.fsum(float(x) * float(y) for x, y in zip(row, right[:, j]))
                for j in range(n)
            ]
            for row in left
        ]
    )
    err = actual - nominal
    l2 = float(np.linalg.norm(err)) / max(float(np.linalg.norm(nominal)), 1e-30)
    peak = float(np.max(np.abs(err))) / max(float(np.max(np.abs(nominal))), 1e-30)
    assert np.all(np.isfinite(actual)) and l2 <= 0.015 and peak <= 0.02
    return dict(
        contract="device-matmul-half-normwise-v1",
        fixed_accuracy_passed=True,
        relative_l2=l2,
        peak_scaled_error=peak,
        per_component_accuracy_not_implied=True,
    )
