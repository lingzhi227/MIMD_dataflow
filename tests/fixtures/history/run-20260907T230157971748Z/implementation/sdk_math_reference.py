"""Source-derived SDK2.10.1 IEEE-half math model; default non-contracted operators.

Derived from preserved math.csl and internal.csl definitions. Sqrt requires
nonnegative finite half operands; exp/SiLU cover finite IEEE-half operands.
Exact SDK approximation behavior does not imply standard all-domain accuracy.
The SDK internal.fmac helper is an ordinary x+y*z expression, NOT @fmach.
"""

import math
import struct
from binary16 import bits, quantize
from frontend import check


def _half(value):
    value = float(value)
    check(
        math.isfinite(value) and value >= 0, "SDK half sqrt finite nonnegative domain"
    )
    rounded = quantize(value)
    check(rounded == value, "SDK half sqrt requires a binary16 operand")
    return rounded


def _add_product(x, y, z):
    return quantize(x + quantize(y * z))


def sqrt_f16(value):
    value = _half(value)
    if value == 0:
        return value
    mantissa, exponent = math.frexp(value)
    exponent -= 1
    normalized = quantize(mantissa * 2)
    if exponent & 1:
        exponent -= 1
        normalized = quantize(normalized * 2)
    guess_bits = 0x59BC - (bits(normalized) >> 1)
    result = struct.unpack("<e", struct.pack("<H", guess_bits))[0]
    z = quantize(-0.5 * normalized)
    for _ in range(2):
        square = quantize(result * result)
        negative_error = _add_product(0.5, z, square)
        result = _add_product(result, result, negative_error)
    result = quantize(result * normalized)
    negative_error = _add_product(-normalized, result, result)
    result = _add_product(result, negative_error, -0.25)
    return quantize(math.ldexp(result, exponent // 2))


def sqrt_reciprocal_f16(value):
    root = sqrt_f16(value)
    if root == 0:
        return math.copysign(math.inf, root)
    return quantize(1.0 / root)


def rms_inverse_f16(square_sum, dimension, epsilon=1e-6):
    square_sum = _half(square_sum)
    check(
        type(dimension) is int and dimension > 0 and quantize(dimension) == dimension,
        "Exact positive half normalization dimension",
    )
    check(math.isfinite(epsilon) and epsilon > 0, "Positive finite RMS epsilon")
    mean = quantize(square_sum / dimension)
    argument = quantize(mean + quantize(epsilon))
    check(argument > 0 and math.isfinite(argument), "Finite positive RMS sqrt argument")
    return sqrt_reciprocal_f16(argument)


def _half_result(value):
    try:
        return quantize(value)
    except OverflowError:
        return math.copysign(math.inf, value)


def exp_f16(value):
    """SDK exp over the observed finite binary16 domain; positive overflow is Inf."""
    from float32 import f32

    value = float(value)
    check(
        math.isfinite(value) and quantize(value) == value,
        "SDK exp requires a finite binary16 operand",
    )
    halfword = lambda word: struct.unpack("<e", struct.pack("<H", word))[0]
    word = lambda value: struct.unpack("<f", struct.pack("<I", value))[0]
    if value > halfword(0x498C):
        return math.inf
    if value < halfword(0xCC55):
        return 0.0
    n = round(f32(value * word(0x40B8AA3B)))
    remainder = quantize(f32(value + f32(n * word(0xBE317218))))
    half_poly = quantize(1 + quantize(0.5 * remainder))
    poly = f32(1 + f32(half_poly * remainder))
    j = n & 3
    exponent = (n - j) >> 2
    table = (0x3F800000, 0x3F9837F0, 0x3FB504F3, 0x3FD744FD)
    expn = f32(math.ldexp(word(table[j]), exponent))
    return _half_result(f32(expn * poly))


def exp_f16_nonpositive(value):
    """Keep the existing strict nonpositive API for stable softmax."""
    check(float(value) <= 0, "SDK exp requires finite nonpositive binary16 operand")
    return exp_f16(value)


def silu_f16(value):
    """Original Prefill half expression, including signed zero and lost tails.

    Exact source arithmetic is not an all-domain standard-accuracy promise.
    """
    value = float(value)
    check(
        math.isfinite(value) and quantize(value) == value,
        "SDK SiLU finite half operand",
    )
    return _half_result(value / _half_result(1 + exp_f16(-value)))


def stable_silu_f16(value):
    """Stable-sign half formulation; finite range is not relative accuracy.

    Keeping the exponent nonpositive avoids positive-exp overflow. Half exp
    underflow can still remove negative tails, so applications need their own
    original-input accuracy gates.
    """
    value = float(value)
    check(math.isfinite(value) and quantize(value) == value, "stable SiLU half input")
    exponential = exp_f16(-abs(value))
    denominator = _half_result(1.0 + exponential)
    numerator = value if value >= 0 else _half_result(value * exponential)
    return _half_result(numerator / denominator)
