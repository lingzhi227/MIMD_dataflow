"""Independent pair-rotation application fixtures and cancellation-aware checks."""

import math
import numpy as np


def batches(m, n, broadcast=False, order="even_odd"):
    rng = np.random.default_rng(210106)
    shape = (1 if broadcast else m, n // 2)
    x = rng.uniform(-2, 2, (m, n))
    angles = np.arange(np.prod(shape)).reshape(shape) * 0.071
    records = [
        (x, np.ones(shape), np.zeros(shape)),
        (x, np.zeros(shape), np.ones(shape)),
        (x, np.cos(angles), np.sin(angles)),
        (np.zeros((m, n)), np.cos(angles), np.sin(angles)),
        (np.full((m, n), 8.0), np.full(shape, 0.5), np.full(shape, 0.5)),
        (rng.uniform(-8, 8, (m, n)), np.cos(angles + 0.37), np.sin(angles + 0.37)),
    ]
    return [
        dict(
            x=np.asarray(a, np.float16).astype(float).ravel().tolist(),
            cosine=np.asarray(c, np.float16).astype(float).ravel().tolist(),
            sine=np.asarray(s, np.float16).astype(float).ravel().tolist(),
        )
        for a, c, s in records
    ]


def check(m, n, broadcast, order, batch, output):
    assert set(batch) == {"x", "cosine", "sine"} and set(output) == {"rotated"}
    x = np.asarray(batch["x"], float).reshape(m, n)
    shape = (1 if broadcast else m, n // 2)
    c = np.asarray(batch["cosine"]).reshape(shape)
    s = np.asarray(batch["sine"]).reshape(shape)
    actual = np.asarray(output["rotated"]).reshape(m, n)
    assert np.all(np.isfinite(actual))
    maxerror = maxratio = 0.0
    for i in range(m):
        for j in range(n // 2):
            a, b = (
                (x[i, 2 * j], x[i, 2 * j + 1])
                if order == "even_odd"
                else (x[i, 2 * j + 1], x[i, 2 * j])
            )
            cc = float(c[0 if broadcast else i, j])
            ss = float(s[0 if broadcast else i, j])
            for k, p, q in [(0, a * cc, -b * ss), (1, b * cc, a * ss)]:
                expected = math.fsum([float(p), float(q)])
                allowance = 0.0015 * (abs(p) + abs(q)) + 2**-23
                error = abs(actual[i, 2 * j + k] - expected)
                assert (
                    error <= allowance
                ), "pair rotation independent component accuracy"
                maxerror = max(maxerror, float(error))
                maxratio = max(maxratio, float(error / allowance))
    return dict(
        contract="pair-rotation-half-v1",
        fixed_accuracy_passed=True,
        max_abs_error=maxerror,
        max_error_over_allowance=maxratio,
        criterion=".0015*(abs(product0)+abs(product1))+2^-23 per output; cancellation-aware",
    )
