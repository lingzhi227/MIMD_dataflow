"""Conservative normal-domain f32 RMS range, including half output rounding.

The pinned SDK sqrt_f32 normal-domain relative-error allowance is 2^-16;
its remaining proof obligations are documented in F32-PRECISION-BOUNDS.md. This
is a range certificate, not a replacement for application accuracy checks.
"""

import math
import numpy as np
from frontend import check

U = 2**-24
ETA = 2**-126  # conservative absolute allowance, including flushed subnormals
SQRT_RELATIVE_ERROR = 2**-16


def upper_f32(value):
    check(
        math.isfinite(value) and 0 <= value <= np.finfo(np.float32).max,
        "finite nonnegative f32 range",
    )
    rounded = np.float32(value)
    if float(rounded) < value:
        rounded = np.nextafter(rounded, np.float32(np.inf))
    check(np.isfinite(rounded), "finite outward f32 bound")
    return float(rounded)


def half_output(gamma, n, epsilon, depth):
    eps = float(np.float32(epsilon))
    check(
        gamma >= 0 and math.isfinite(gamma) and type(n) is int and 1 <= n <= 2048,
        "f32 RMS range parameters",
    )
    check(
        type(depth) is int
        and n <= depth <= 4096
        and math.isfinite(eps)
        and eps >= 2**-24,
        "normal-domain RMS epsilon/depth",
    )
    alpha = (1 - U) ** (depth + 3)
    floor = (1 - U) * eps - (4 + depth) * ETA
    inv = (1 + U) / (math.sqrt(floor) * (1 - SQRT_RELATIVE_ERROR))
    limit = (
        gamma * math.sqrt(n / alpha) * (1 + U) ** 3 / (1 - SQRT_RELATIVE_ERROR)
        + ETA * (1 + U) * inv
        + ETA
    )
    check(limit <= 65504, "f32 RMS may overflow half result")
    value = np.float16(limit)
    if float(value) < limit:
        value = np.nextafter(value, np.float16(np.inf))
    return dict(
        output=float(value),
        pre_storage_upper=limit,
        positive_sum_depth=depth,
        sdk_sqrt_relative_error_allowance=SQRT_RELATIVE_ERROR,
        reciprocal_upper=inv,
        epsilon_f32=eps,
        scope="Nonnegative f32 square/sum/divide, epsilon floor and correlated row norm; SDK normal-domain sqrt contract, gamma-first f32 operations then half storage. Range only.",
    )


def contraction(l1_times_operand, n):
    check(
        l1_times_operand >= 0 and math.isfinite(l1_times_operand) and 1 <= n <= 2048,
        "f32 contraction range",
    )
    growth = (1 + U) ** n
    return upper_f32(l1_times_operand * growth + ETA * (growth - 1) / U)
