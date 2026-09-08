"""Independent standard gated-SiLU fixtures; account for stored half intermediates."""



import math
import numpy as np


def batches(m, n):
    rng = np.random.default_rng(210104)
    size = m * n
    up = rng.uniform(-2, 2, (m, n))
    gate = rng.uniform(-8, 8, (m, n))
    pattern = np.resize(
        np.array([-8.0, -1.0, -(2**-24), 0.0, 2**-24, 1.0, 8.0]), size
    ).reshape(m, n)
    tiny = np.resize(
        np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]) * 2**-24, size
    ).reshape(m, n)
    values = [
        (up, gate),
        (np.resize(np.array([-0.5, 1.0, -2.0, 0.25]), size).reshape(m, n), pattern),
        (np.zeros((m, n)), gate),
        (np.full((m, n), 8.0), pattern),
        (np.resize(np.array([-8.0, 8.0, -1.0, 1.0]), size).reshape(m, n), tiny),
        (rng.uniform(-8, 8, (m, n)), rng.uniform(-8, 8, (m, n))),
    ]
    return [
        dict(
            up=np.asarray(a, np.float16).astype(float).ravel().tolist(),
            gate=np.asarray(b, np.float16).astype(float).ravel().tolist(),
        )
        for a, b in values
    ]


def check(m, n, batch, output):
    assert set(batch) == {"up", "gate"} and set(output) == {"gated"}
    up = np.asarray(batch["up"], float).reshape(m, n)
    gate = np.asarray(batch["gate"], float).reshape(m, n)
    actual = np.asarray(output["gated"], float).reshape(m, n)
    assert all(np.all(np.isfinite(a)) for a in (up, gate, actual))
    reference = np.empty_like(gate)
    for i, (u, g) in enumerate(zip(up.ravel(), gate.ravel())):
        e = math.exp(-abs(float(g)))
        v = float(g) / (1 + e) if g >= 0 else float(g) * e / (1 + e)
        reference.ravel()[i] = float(u) * v
    error = np.abs(actual - reference)
    allowance = 0.004 * np.abs(reference) + 2**-24 * (1 + np.abs(up))
    assert np.all(
        error <= allowance
    ), "gated SiLU componentwise half accuracy with subnormal rounding allowance"
    return dict(
        contract="gated-activation-half-v1",
        fixed_accuracy_passed=True,
        max_abs_error=float(error.max()),
        max_error_over_allowance=float(np.max(error / allowance)),
        componentwise_relative_term=0.004,
        absolute_rounding_term="2^-24*(1+abs(up))",
        reference="independent stable math.exp SiLU times up; unrounded mathematical composition",
    )
