"""Original-input independent attention arithmetic and changed warm inputs."""



import math
import numpy as np


def batches(m, n):
    rng = np.random.default_rng(210108)
    pairs = [
        (np.eye(m, n), np.roll(np.eye(m, n), 3, axis=0)),
        (rng.uniform(-0.25, 0.25, (m, n)), rng.uniform(-0.25, 0.25, (m, n))),
        (np.zeros((m, n)), rng.uniform(-0.25, 0.25, (m, n))),
    ]
    triples = [(q, k, rng.uniform(-0.25, 0.25, (m, n))) for q, k in pairs]
    triples.extend(
        [
            (
                np.full((m, n), 1 / 16),
                np.full((m, n), -1 / 16),
                rng.uniform(-0.25, 0.25, (m, n)),
            ),
            (
                rng.uniform(-0.25, 0.25, (m, n)),
                rng.uniform(-0.25, 0.25, (m, n)),
                np.zeros((m, n)),
            ),
            (
                rng.uniform(-0.5, 0.5, (m, n)),
                rng.uniform(-0.5, 0.5, (m, n)),
                rng.uniform(-0.25, 0.25, (m, n)),
            ),
        ]
    )
    return [
        {
            name: np.asarray(a, np.float16).astype(float).ravel().tolist()
            for name, a in zip(("q", "k", "v"), triple)
        }
        for triple in triples
    ]


def check(m, n, scale, b, o):
    assert set(b) == {"q", "k", "v"} and set(o) == {"output"}
    q, k, v = [np.asarray(b[name]).reshape(m, n) for name in ("q", "k", "v")]
    actual = np.asarray(o["output"]).reshape(m, n)
    logits = [
        [math.fsum(float(x) * float(y) for x, y in zip(row, col)) * scale for col in k]
        for row in q
    ]
    prob = []
    for row in logits:
        e = [math.exp(x - max(row)) for x in row]
        total = math.fsum(e)
        prob.append([x / total for x in e])
    nominal = np.asarray(
        [
            [math.fsum(float(x) * float(y) for x, y in zip(row, col)) for col in v.T]
            for row in prob
        ]
    )
    err = actual - nominal
    l2 = float(np.linalg.norm(err)) / max(float(np.linalg.norm(nominal)), 1e-30)
    peak = float(np.max(np.abs(err))) / max(float(np.max(np.abs(nominal))), 1e-30)
    assert np.all(np.isfinite(actual)) and l2 <= 0.02 and peak <= 0.025
    return dict(
        contract="attention-half-normwise-v1",
        fixed_accuracy_passed=True,
        relative_l2=l2,
        peak_scaled_error=peak,
        per_component_accuracy_not_implied=True,
    )
