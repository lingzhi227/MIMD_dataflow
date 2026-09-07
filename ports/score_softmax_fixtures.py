"""Independent original-input scaled attention-score normalization checks."""

import math
import numpy as np
from score_fixtures import batches


def check(m, n, scale, b, o):
    assert set(b) == {"q", "k"} and set(o) == {"probability"}
    q = np.asarray(b["q"]).reshape(m, n)
    k = np.asarray(b["k"]).reshape(m, n)
    actual = np.asarray(o["probability"]).reshape(m, m)
    logits = [
        [
            math.fsum(float(x) * float(y) for x, y in zip(qrow, krow)) * scale
            for krow in k
        ]
        for qrow in q
    ]
    nominal = []
    for row in logits:
        peak = max(row)
        exp = [math.exp(x - peak) for x in row]
        total = math.fsum(exp)
        nominal.append([x / total for x in exp])
    nominal = np.asarray(nominal)
    err = actual - nominal
    l2 = float(np.linalg.norm(err)) / float(np.linalg.norm(nominal))
    peak = float(np.max(np.abs(err))) / float(np.max(np.abs(nominal)))
    mass = float(np.max(np.abs(actual.sum(axis=1) - 1)))
    assert (
        np.all(np.isfinite(actual))
        and np.all(actual >= 0)
        and l2 <= 0.015
        and peak <= 0.02
        and mass <= 0.01
    )
    return dict(
        contract="score-softmax-half-normwise-v1",
        fixed_accuracy_passed=True,
        relative_l2=l2,
        peak_scaled_error=peak,
        max_row_mass_error=mass,
        per_component_accuracy_not_implied=True,
    )
