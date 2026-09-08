"""Componentwise f32 dot-product error contract, not a numerical code generator.

Uses a conservative gamma_(2K+1) envelope for multiply/add or FMA evaluation.
See docs/NUMERICAL-POLICY.md for assumptions, sources and scope.
"""


def check_matrix_roundoff(a, b, actual, reference, *, reference_is_f32=False):
    import numpy as np

    a, b, actual, reference = [
        np.asarray(v, np.float64) for v in (a, b, actual, reference)
    ]
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1] != b.shape[0]
        or actual.shape != (a.shape[0], b.shape[1])
        or reference.shape != actual.shape
    ):
        raise ValueError("roundoff matrix shapes")
    if not all(np.all(np.isfinite(v)) for v in (a, b, actual, reference)):
        raise ValueError("roundoff requires finite data")
    if any(
        not np.array_equal(v, v.astype(np.float32).astype(np.float64)) for v in (a, b)
    ):
        raise ValueError("roundoff inputs must be f32 values")
    tiny = np.finfo(np.float32).tiny
    if any(np.any((np.abs(v) > 0) & (np.abs(v) < tiny)) for v in (a, b)):
        raise ValueError("roundoff subnormal inputs unsupported (DAZ)")
    k = a.shape[1]
    operations = 2 * k + 1
    u = 2.0**-24
    if not 0 < operations * u < 1:
        raise ValueError("roundoff reduction length")
    gamma = operations * u / (1 - operations * u)
    magnitude = np.abs(a) @ np.abs(b)
    if np.any(magnitude > np.finfo(np.float32).max):
        raise ValueError("roundoff no-overflow contract")
    # Covers lost subnormal terms conservatively if a target flushes them.
    tiny_allowance = operations * tiny / (1 - operations * u)
    reference_gamma = operations * 2.0**-53 / (1 - operations * 2.0**-53)
    factor = 2 if reference_is_f32 else 1
    budget = np.nextafter(
        (factor * gamma + reference_gamma) * magnitude + factor * tiny_allowance, np.inf
    )
    error = np.abs(actual - reference)
    ratio = error / budget
    report = {
        "policy": "componentwise-f32-dot-v1",
        "K": k,
        "operations_bound": operations,
        "gamma": gamma,
        "reference_is_f32": reference_is_f32,
        "max_abs_error": float(np.max(error)),
        "max_error_over_bound": float(np.max(ratio)),
        "old_fixed_tolerance_passed": bool(
            np.allclose(actual, reference, rtol=3e-5, atol=3e-6)
        ),
    }
    if np.any(error > budget):
        raise ValueError("dot-product rounding envelope exceeded: " + str(report))
    return report
