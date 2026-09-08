"""Original-input application math, separate from compiler and CSL target models."""



import math
import numpy as np


def batches(m, n):
    rng = np.random.default_rng(210112)
    out = []
    half = lambda v: np.asarray(v, np.float16).astype(float)
    for epoch in range(8):
        a = half(rng.uniform(-0.125, 0.125, (m, n)))
        w = half(rng.uniform(-0.125, 0.125, (n, n)))
        r = half(rng.uniform(-0.5, 0.5, (m, n)))
        r = half(r * np.linspace(0.25, 1, m)[:, None])
        g = half(rng.uniform(0.5, 1.5, (1, n)))
        if epoch == 2:
            a.fill(0)
            r.fill(0)
        if epoch == 3:
            w.fill(0)
        if epoch == 4:
            r.fill(0)
        if epoch == 5:
            a.fill(0.125)
            w.fill(0.125)
            r.fill(0.5)
            g.fill(1.5)
        if epoch == 6:
            a = rng.integers(-8, 9, (m, n)).astype(float) / 64
            w = np.eye(n) * 0.125
            r = -a * 0.125
        if epoch == 7:
            w.fill(0)
            r.fill(0)
        out.append(
            {
                k: v.ravel().tolist()
                for k, v in zip(
                    ("activation", "weight", "residual", "gamma"), (a, w, r, g)
                )
            }
        )
    return out


def check(m, n, epsilon, batch, output):
    a = np.asarray(batch["activation"]).reshape(m, n)
    w = np.asarray(batch["weight"]).reshape(n, n)
    r = np.asarray(batch["residual"]).reshape(m, n)
    g = np.asarray(batch["gamma"]).reshape(1, n)
    projected = np.asarray(
        [
            [math.fsum(float(x) * float(y) for x, y in zip(row, col)) for col in w.T]
            for row in a
        ]
    )
    z = projected + r
    means = np.asarray([math.fsum(float(x) * float(x) for x in row) / n for row in z])
    expected = z * g / np.sqrt(means[:, None] + epsilon)
    assert set(output) == {"output"}
    actual = np.asarray(output["output"]).reshape(m, n)
    error = actual - expected
    l2 = float(np.linalg.norm(error)) / max(float(np.linalg.norm(expected)), 1e-30)
    peak = float(np.max(np.abs(error))) / max(float(np.max(np.abs(expected))), 1e-30)
    assert np.all(np.isfinite(actual)) and l2 <= 0.02 and peak <= 0.03, (l2, peak)
    return dict(
        contract="projection-residual-rms-half-normwise-v1",
        fixed_accuracy_passed=True,
        relative_l2=l2,
        peak_scaled_error=peak,
        per_component_accuracy_not_implied=True,
    )
