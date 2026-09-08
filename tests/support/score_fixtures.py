"""Independent original-input score fixtures and math.fsum accuracy review."""



import math
import numpy as np


def batches(m, n):
    rng = np.random.default_rng(210108)
    pairs = [
        (np.eye(m, n), np.roll(np.eye(m, n), 3, axis=0)),
        (rng.uniform(-0.25, 0.25, (m, n)), rng.uniform(-0.25, 0.25, (m, n))),
        (np.zeros((m, n)), rng.uniform(-0.25, 0.25, (m, n))),
        (np.full((m, n), 1 / 16), np.full((m, n), -1 / 16)),
        (rng.uniform(-0.25, 0.25, (m, n)), rng.uniform(-0.25, 0.25, (m, n))),
        (rng.uniform(-0.5, 0.5, (m, n)), rng.uniform(-0.5, 0.5, (m, n))),
    ]
    return [
        dict(
            q=np.asarray(q, np.float16).astype(float).ravel().tolist(),
            k=np.asarray(k, np.float16).astype(float).ravel().tolist(),
        )
        for q, k in pairs
    ]


def check(m, n, b, o):
    assert set(b) == {"q", "k"} and set(o) == {"score"}
    q = np.asarray(b["q"]).reshape(m, n)
    k = np.asarray(b["k"]).reshape(m, n)
    actual = np.asarray(o["score"]).reshape(m, m)
    nominal = np.array(
        [
            [math.fsum(float(x) * float(y) for x, y in zip(qrow, krow)) for krow in k]
            for qrow in q
        ]
    )
    err = actual - nominal
    l2 = float(np.linalg.norm(err)) / max(float(np.linalg.norm(nominal)), 1e-30)
    peak = float(np.max(np.abs(err))) / max(float(np.max(np.abs(nominal))), 1e-30)
    assert np.all(np.isfinite(actual)) and l2 <= 0.015 and peak <= 0.02
    return dict(
        contract="score-half-normwise-v1",
        fixed_accuracy_passed=True,
        relative_l2=l2,
        peak_scaled_error=peak,
        per_component_accuracy_not_implied=True,
    )
