"""Independent application reference for resident normalization/projection."""



import math
import numpy as np
from rms_fixtures import batches as rms_batches


def batches(m, n):
    values = rms_batches(m, n)
    rng = np.random.default_rng(210103)
    for epoch, v in enumerate(values):
        q = np.eye(n) if epoch in (0, 4) else rng.uniform(-1 / 16, 1 / 16, (n, n))
        v["q"] = np.asarray(q, np.float16).astype(float).ravel().tolist()
    return values


def check(m, n, batch, output):
    assert set(batch) == {"x", "w", "q"} and set(output) == {"projected"}
    x = np.asarray(batch["x"], float).reshape(m, n)
    w = np.asarray(batch["w"], float)
    q = np.asarray(batch["q"], float).reshape(n, n)
    actual = np.asarray(output["projected"], float).reshape(m, n)
    assert all(np.all(np.isfinite(a)) for a in (x, w, q, actual))
    normalized = np.empty((m, n))
    for i, row in enumerate(x):
        divisor = math.sqrt(math.fsum(float(v) ** 2 for v in row) / n + 1e-6)
        normalized[i] = [
            float(v) * float(weight) / divisor for v, weight in zip(row, w)
        ]
    # Standard mathematical composition; no target helper, half FMA, SDK math,
    # compiler IR, or communication order used by this independent reference.
    reference = np.empty((m, n))
    for i, row in enumerate(normalized):
        for j, column in enumerate(q.T):
            reference[i, j] = math.fsum(
                float(a) * float(b) for a, b in zip(row, column)
            )
    error = actual - reference
    relative = float(np.linalg.norm(error)) / max(
        float(np.linalg.norm(reference)), 1e-30
    )
    maximum = float(np.max(np.abs(error)))
    peak = float(np.max(np.abs(reference)))
    assert relative <= 0.015 and maximum <= 0.02 * max(
        peak, 1e-30
    ), "resident normalization/projection standard accuracy"
    return dict(
        contract="normalized-matmul-half-normwise-v1",
        fixed_accuracy_passed=True,
        relative_l2_error=relative,
        max_abs_error=maximum,
        reference="independent math.fsum row RMS and dot products, standard math.sqrt, epsilon 1e-6",
        per_component_accuracy_not_implied=True,
    )
