"""Range proof for the existing half row collective and validated SDK math model."""

from functools import lru_cache
import math, struct
from frontend import check
from binary16 import quantize, bits
from input_contracts import half_product_bound, half_dot_bound, half_operand_bound
from sdk_math_reference import rms_inverse_f16


@lru_cache(maxsize=32)
def inverse_bound(total, dimension, epsilon):
    # Enumerate representable nonnegative reduced sums; do not assume that an
    # approximate SDK sqrt implementation is perfectly monotone.
    maximum = 0.0
    for word in range(bits(total) + 1):
        value = struct.unpack("<e", struct.pack("<H", word))[0]
        maximum = max(maximum, rms_inverse_f16(value, dimension, epsilon))
    check(math.isfinite(maximum), "RMS inverse range must be finite")
    return maximum


@lru_cache(maxsize=32)
def correlated_output_bound(value, gamma, total, dimension, epsilon):
    """Bound the executed gamma-first scale using its own squared contribution.

    Every local/tree sum is a rounded addition of nonnegative half operands,
    so the reduced sum cannot be below any participating rounded square.
    Suffix maxima cover all permitted SDK inverse values above that lower
    bound; no approximate-sqrt monotonicity or real-arithmetic RMS inequality
    is assumed. Squaring underflow is included, as is gamma-first rounding.
    """
    vbound, gbound = half_operand_bound(value), half_operand_bound(gamma)
    half_product_bound(vbound, vbound)
    half_product_bound(vbound, gbound)
    check(
        quantize(total) == total and total >= quantize(vbound * vbound),
        "RMS total must contain each rounded square",
    )
    suffix = [0.0] * (bits(total) + 1)
    maximum = 0.0
    for word in range(bits(total), -1, -1):
        reduced = struct.unpack("<e", struct.pack("<H", word))[0]
        inv = rms_inverse_f16(reduced, dimension, epsilon)
        check(math.isfinite(inv) and inv >= 0, "finite nonnegative SDK inverse")
        maximum = max(maximum, inv)
        suffix[word] = maximum
    output = 0.0
    for word in range(bits(vbound) + 1):
        v = struct.unpack("<e", struct.pack("<H", word))[0]
        square = quantize(v * v)
        weighted = quantize(v * gbound)
        product = weighted * suffix[bits(square)]
        check(product <= 65504, "correlated RMS output may overflow")
        output = max(output, quantize(product))
    return output


def row_norm_bound(value, gamma, local_features, columns, epsilon):
    check(
        columns in (4, 8) and type(local_features) is int and local_features > 0,
        "RMS source row tree range",
    )
    square = half_product_bound(value, value)
    local = half_dot_bound(square, 1.0, local_features)

    def add(a, b):
        check(a + b <= 65504, "RMS row sum may overflow")
        return quantize(a + b)

    left = local
    for _ in range(columns // 2 - 1):
        left = add(left, local)
    right = local
    for _ in range(columns // 2 - 2):
        right = add(right, local)
    total = add(add(local, right), left)
    inverse = inverse_bound(total, local_features * columns, epsilon)
    weighted = half_product_bound(value, gamma)
    output = correlated_output_bound(
        value, gamma, total, local_features * columns, epsilon
    )
    return dict(
        square=square,
        local_square_sum=local,
        reduced_square_sum=total,
        inverse=inverse,
        weighted=weighted,
        output=output,
        proof="Monotone nonnegative source half tree; exhaustive representable reduced-sum SDK inverse model. Each rounded input square bounds its reduced sum from below; inverse suffix maxima retain that correlation through gamma-first rounding. Range safety, not an accuracy guarantee.",
    )
