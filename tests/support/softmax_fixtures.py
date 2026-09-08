"""Stable row softmax fixtures and independent standard-math accuracy checks."""



import math
import numpy as np


def batches(m, n):
    rng = np.random.default_rng(210102)
    x = rng.uniform(-2, 2, (m, n))
    peaks = np.full((m, n), -128.0)
    peaks[np.arange(m), (np.arange(m) * 17 + 3) % n] = 128
    rows = np.repeat((np.arange(m) % 17 - 8)[:, None], n, axis=1).astype(float)
    values = [
        x,
        np.full((m, n), -1024.0),
        np.zeros((m, n)),
        peaks,
        rows,
        rng.uniform(-8, 8, (m, n)),
    ]
    return [
        dict(x=np.asarray(x, np.float16).astype(float).ravel().tolist()) for x in values
    ]


def check(m, n, batch, output):
    assert set(batch) == {"x"} and set(output) == {"probability"}
    x = np.asarray(batch["x"], float).reshape(m, n)
    actual = np.asarray(output["probability"], float).reshape(m, n)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(actual))
    assert np.all(actual >= 0), "probabilities must be nonnegative"
    ref = np.empty((m, n))
    for i, row in enumerate(x):
        scaled = [float(v) * 0.125 for v in row]
        maximum = max(scaled)
        exponents = [math.exp(v - maximum) for v in scaled]
        denominator = math.fsum(exponents)
        ref[i] = [v / denominator for v in exponents]
    error = actual - ref
    norm = math.sqrt(math.fsum(float(v) ** 2 for v in ref.ravel()))
    relative = math.sqrt(math.fsum(float(v) ** 2 for v in error.ravel())) / norm
    maximum = float(np.max(np.abs(error)))
    mass_error = max(abs(math.fsum(map(float, row)) - 1) for row in actual)
    assert relative <= 0.01 and maximum <= 0.015 * float(
        np.max(ref)
    ), "softmax half normwise accuracy"
    assert mass_error <= 0.01, "softmax row probability mass"
    return dict(
        contract="softmax-half-normwise-v1",
        fixed_accuracy_passed=True,
        relative_l2_error=relative,
        max_abs_error=maximum,
        max_row_mass_error=mass_error,
        reference="independent math.exp and math.fsum; stable row softmax, scale 0.125",
        per_component_accuracy_not_implied=True,
    )
