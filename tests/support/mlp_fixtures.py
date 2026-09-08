"""Independent application arithmetic and changed warm inputs for rectangular MLP."""



import math
import numpy as np


def batches(m, n, f, blocks=None):
    rng = np.random.default_rng(210110)
    bs = []
    for epoch in range(6):
        x = rng.uniform(-0.125, 0.125, (m, n))
        if epoch == 2:
            x[:] = 0
        u, g, d = [
            rng.uniform(-0.125, 0.125, shape) for shape in ((n, f), (n, f), (f, n))
        ]
        if epoch == 3:
            x[:] = 0.125
            g[:] = -0.125
        if epoch == 4:
            d[:] = 0
        if epoch == 5:
            x[:] = 0.125
            u[:] = 0.125
            g[:] = 0.125
            d[:] = 0.125
        bs.append(
            {
                k: np.asarray(v, np.float16).astype(float).ravel().tolist()
                for k, v in zip(
                    ("x", "up_weight", "gate_weight", "down_weight"), (x, u, g, d)
                )
            }
        )
    if blocks is not None:
        assert blocks in (4, 8) and f % blocks == 0
        x = np.full((m, n), 0.125)
        u = np.full((n, f), 0.125)
        g = u.copy()
        d = np.zeros((f, n))
        tile = f // blocks
        for pair in range(blocks // 2 - 1):
            d[2 * pair * tile : (2 * pair + 1) * tile] = 0.125 / (2**pair)
            d[(2 * pair + 1) * tile : (2 * pair + 2) * tile] = -0.125 / (2**pair)
        d[(blocks - 2) * tile : (blocks - 1) * tile] = 2**-10
        for values in [
            (x, u, g, d),
            (
                rng.uniform(-0.125, 0.125, (m, n)),
                rng.uniform(-0.125, 0.125, (n, f)),
                rng.uniform(-0.125, 0.125, (n, f)),
                np.zeros((f, n)),
            ),
        ]:
            bs.append(
                {
                    k: np.asarray(v, np.float16).astype(float).ravel().tolist()
                    for k, v in zip(
                        ("x", "up_weight", "gate_weight", "down_weight"), values
                    )
                }
            )
    return bs


def check(m, n, f, b, o):
    x = np.asarray(b["x"]).reshape(m, n)
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

    up = product(x, u)
    gate = product(x, g)
    hidden = np.asarray(
        [
            [float(a) * float(v) / (1 + math.exp(-float(v))) for a, v in zip(row, gr)]
            for row, gr in zip(up, gate)
        ]
    )
    expected = product(hidden, d)
    actual = np.asarray(o["output"]).reshape(m, n)
    err = actual - expected
    l2 = float(np.linalg.norm(err)) / max(float(np.linalg.norm(expected)), 1e-30)
    peak = float(np.max(np.abs(err))) / max(float(np.max(np.abs(expected))), 1e-30)
    assert np.all(np.isfinite(actual)) and l2 <= 0.02 and peak <= 0.03
    return dict(
        contract="rectangular-mlp-half-normwise-v1",
        fixed_accuracy_passed=True,
        relative_l2=l2,
        peak_scaled_error=peak,
        per_component_accuracy_not_implied=True,
    )
