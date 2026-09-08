"""Original-domain row normalization fixtures and independent math.fsum reference."""



import math
import numpy as np


def batches(m, n):
    rng = np.random.default_rng(210101)
    weights = (np.arange(n) % 17 - 8) / 8
    pattern = np.tile(np.array([-0.5, 0.25, 0.5, -0.25]), (n + 3) // 4)[:n]
    scales = (1 + np.arange(m) % 16) / 16
    x = rng.uniform(-0.5, 0.5, (m, n))
    values = [
        (x, weights),
        (scales[:, None] * pattern, weights[::-1]),
        (-x, np.ones(n)),
        (np.zeros((m, n)), weights),
        (np.full((m, n), 2**-12), weights),
        (rng.uniform(-1, 1, (m, n)), rng.uniform(-1, 1, n)),
    ]
    return [
        dict(
            x=np.asarray(x, np.float16).astype(float).ravel().tolist(),
            w=np.asarray(w, np.float16).astype(float).tolist(),
        )
        for x, w in values
    ]


def check(m, n, batch, output):
    assert set(batch) == {"x", "w"} and set(output) == {"normalized"}
    x = np.asarray(batch["x"], float).reshape(m, n)
    w = np.asarray(batch["w"], float).reshape(1, n)
    actual = np.asarray(output["normalized"], float).reshape(m, n)
    assert (
        np.all(np.isfinite(x))
        and np.all(np.isfinite(w))
        and np.all(np.isfinite(actual))
    )
    ref = np.empty((m, n))
    for i in range(m):
        denominator = math.sqrt(math.fsum(float(v) * float(v) for v in x[i]) / n + 1e-6)
        ref[i] = [
            float(v) * float(weight) / denominator for v, weight in zip(x[i], w[0])
        ]
    error = actual - ref
    norm = math.sqrt(math.fsum(float(v) * float(v) for v in ref.ravel()))
    enorm = math.sqrt(math.fsum(float(v) * float(v) for v in error.ravel()))
    relative = enorm / norm if norm else enorm
    peak = float(np.max(np.abs(ref)))
    maximum = float(np.max(np.abs(error)))
    assert relative <= 0.01 and maximum <= 0.015 * max(
        peak, 1e-30
    ), "standard RMSNorm half policy accuracy"
    return dict(
        contract="rms-half-normwise-v1",
        fixed_accuracy_passed=True,
        relative_l2_error=relative,
        max_abs_error=maximum,
        reference="independent math.fsum square norm and standard math.sqrt; epsilon1e-6",
        per_component_accuracy_not_implied=True,
    )
