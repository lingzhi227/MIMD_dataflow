"""Correlated row-L1 bounds for projections consuming the SDK half RMS region.

This supports future resident connections; it does not authorize a lowering or
replace original-input accuracy gates. Existing profiles do not select it.
"""

import functools, math, struct
import numpy as np
from frontend import check
from binary16 import bits
from input_contracts import half_operand_bound
from rms_bounds import row_norm_bound
from sdk_math_reference import rms_inverse_f16


def outward_half(value):
    check(math.isfinite(value) and 0 <= value < 65504, "finite half range certificate")
    if value == 0:
        return 0.0
    rounded = np.float16(value)
    check(float(rounded) < 65504, "finite successor required for outward half bound")
    result = float(np.nextafter(rounded, np.float16(math.inf)))
    check(math.isfinite(result), "outward half certificate must stay finite")
    return result


@functools.lru_cache(maxsize=32)
def normalized_l1(value, gamma, local_features, columns, epsilon):
    """Bound a row, retaining its squared-sum correlation through SDK inverse.

    Positive additions of representable half operands are exact below the
    normal interval. Therefore their relative lower bound holds even when
    individual squares underflow; a separate eta term bounds those squares.
    Enumerating the actual inverse model avoids a monotonic-sqrt assumption.
    """
    v, g = half_operand_bound(value), half_operand_bound(gamma)
    limits = row_norm_bound(v, g, local_features, columns, epsilon)
    n = local_features * columns
    check(1 <= n <= 512, "bounded RMS-to-projection row")
    u, eta = 2**-11, 2**-25
    depth = n + columns
    lower = (1 - u) ** depth
    target = 0.0
    witness = 0.0
    for word in range(bits(limits["reduced_square_sum"]) + 1):
        reduced = struct.unpack("<e", struct.pack("<H", word))[0]
        real_square_sum = min(n * v * v, (reduced / lower + n * eta) / (1 - u))
        input_l1 = math.sqrt(n * real_square_sum)
        inverse = rms_inverse_f16(reduced, n, epsilon)
        candidate = (1 + u) * inverse * ((1 + u) * g * input_l1 + n * eta) + n * eta
        if candidate > target:
            target, witness = candidate, reduced
    # Native RMS evaluates the original row in double, then rounds each output.
    # The small double margin dominates its <=512-term positive accumulation.
    native = n * g * (1 + u) * (1 + 2**-40) + n * eta
    if v == 0 or g == 0:
        target = native = 0.0
    return dict(
        target_l1_upper=target,
        native_l1_upper=native,
        row_l1_upper=outward_half(max(target, native)),
        inverse_witness_sum=witness,
        reduced_sum_upper=limits["reduced_square_sum"],
        positive_sum_depth=depth,
        enumerated_sums=bits(limits["reduced_square_sum"]) + 1,
        dimension=n,
        half_unit_roundoff=u,
        half_underflow_absolute=eta,
        scope="Correlated row range using pinned SDK inverse model and declared source half row reduction; not an accuracy or arbitrary-CSL proof",
    )


def projection(value, gamma, weight, local_features, columns, epsilon):
    certificate = normalized_l1(value, gamma, local_features, columns, epsilon)
    w = half_operand_bound(weight)
    n = certificate["dimension"]
    u, eta = 2**-11, 2**-25
    growth = (1 + u) ** n
    raw = growth * w * certificate["row_l1_upper"] + eta * (growth - 1) / u
    if w == 0 or certificate["row_l1_upper"] == 0:
        raw = 0.0
    return dict(
        normalized_row=certificate,
        weight_absolute=w,
        projection_absolute=outward_half(raw),
        half_fma_depth=n,
        scope="Any permutation of N ordinary half FMA contributions; block-f32 policy requires its own bound",
    )
