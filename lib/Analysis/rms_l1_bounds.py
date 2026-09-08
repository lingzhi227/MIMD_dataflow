"""Cauchy-Schwarz range propagation for half RMS with SDK f32 reduction.

This is a range proof, not an accuracy guarantee. It retains the shared row
normalization correlation instead of pretending every feature can independently
attain the elementwise maximum. Target arithmetic never consumes this model.
"""

from functools import lru_cache
from fractions import Fraction
import math, struct
import numpy as np
from frontend import check
from binary16 import bits, quantize
from input_contracts import half_operand_bound, half_product_bound, half_dot_bound
from sdk_math_reference import rms_inverse_f16
from rms_bounds import correlated_output_bound

U = Fraction(1, 2048)
U32 = Fraction(1, 16777216)
ETA = Fraction(1, 16777216)


def upward(value):
    """Binary64 upper enclosure of a nonnegative exact rational."""
    assert value >= 0
    f = float(value)
    if Fraction.from_float(f) < value:
        f = math.nextafter(f, math.inf)
    return f


def half_ceiling(value):
    check(math.isfinite(value) and 0 <= value <= 65504, "finite half range enclosure")
    q = np.float16(value)
    if float(q) < value:
        q = np.nextafter(q, np.float16(math.inf))
    check(np.isfinite(q), "finite half ceiling")
    return float(q)


@lru_cache(maxsize=32)
def normalized_l1(value, gamma, total, dimension, local_features, columns, epsilon):
    check(
        type(columns) is int
        and type(dimension) is int
        and columns in (4, 8, 16)
        and type(local_features) is int
        and 1 <= local_features <= 512
        and dimension == columns * local_features,
        "SDK RMS row geometry",
    )
    v = Fraction(half_operand_bound(value))
    g = Fraction(half_operand_bound(gamma))
    check(
        0 <= total <= 65504 and float(np.float16(total)) == total,
        "representable reduced RMS bound",
    )
    check(
        math.isfinite(epsilon)
        and 0 < float(np.float16(epsilon)) <= 1
        and dimension <= 2048,
        "SDK RMS epsilon/dimension",
    )
    square = half_product_bound(float(v), float(v))
    local = half_dot_bound(square, 1.0, local_features)
    half_product_bound(float(v), float(g))
    check(
        local * columns <= 65504 and total >= quantize(local * columns),
        "RMS enumeration must cover the independently bounded total",
    )
    # The rounding envelope below assumes finite normalized values. Retain the
    # independent elementwise proof before using Cauchy-Schwarz to tighten L1.
    elementwise = correlated_output_bound(float(v), float(g), total, dimension, epsilon)
    # q(x*x), at most Nt local additions, then at most P-1 f32 additions and
    # one final half narrowing: S >= A*T - B. Gradual-underflow error eta/2
    # is retained even though local positive half additions need no such term.
    a = (1 - U) ** (local_features + 2) * (1 - U32) ** (columns - 1)
    loss = dimension * ETA + ETA / 2
    maximum = 0.0
    witness = 0.0
    for word in range(bits(total) + 1):
        reduced = struct.unpack("<e", struct.pack("<H", word))[0]
        inverse = Fraction(rms_inverse_f16(reduced, dimension, epsilon))
        t = min(dimension * v * v, (Fraction(reduced) + loss) / a)
        # Cauchy-Schwarz sum|x| <= sqrt(N*T); outward-round the sqrt input,
        # sqrt result, and final exact-rational arithmetic enclosure.
        radicand = dimension * t
        root = math.sqrt(upward(radicand))
        while Fraction.from_float(root) ** 2 < radicand:
            root = math.nextafter(root, math.inf)
        bound = (1 + U) ** 2 * g * inverse * Fraction(root) + dimension * (
            (1 + U) * ETA / 2 * inverse + ETA / 2
        )
        upper = upward(bound)
        if upper > maximum:
            maximum = upper
            witness = reduced
    factor_lower = float(a)
    if Fraction.from_float(factor_lower) > a:
        factor_lower = math.nextafter(factor_lower, -math.inf)
    return dict(
        bound=maximum,
        elementwise_bound=elementwise,
        reduced_sum_witness=witness,
        square_sum_lower_factor=factor_lower,
        square_sum_lower_factor_exact=[str(a.numerator), str(a.denominator)],
        underflow_loss=upward(loss),
        proof="S >= (1-u16)^(Nt+2)*(1-u32)^(P-1)*T - (N+1/2)*eta16; Cauchy-Schwarz; enumerate permitted half S using pinned SDK inverse. Gamma-first and final half rounding retain relative and gradual-underflow terms. Exact rational arithmetic with outward binary64 sqrt enclosure.",
    )


def projection(l1, weight, local_features, columns):
    check(
        type(local_features) is int
        and 1 <= local_features <= 512
        and type(columns) is int
        and columns in (4, 8, 16),
        "SDK local projection geometry",
    )
    check(math.isfinite(l1) and l1 >= 0, "finite normalized L1 bound")
    k = (1 + U) ** local_features
    # Bound each local FMA chain using the global L1 bound. For the global
    # reduction, sum local L1 values only once; do not multiply it by P.
    local = k * (
        Fraction(l1) * Fraction(half_operand_bound(weight)) + local_features * ETA / 2
    )
    merged = (
        (1 + U32) ** (columns - 1)
        * k
        * (
            Fraction(l1) * Fraction(half_operand_bound(weight))
            + columns * local_features * ETA / 2
        )
    )
    final = (1 + U) * merged + ETA / 2
    return dict(
        local=half_ceiling(upward(local)),
        reduced=half_ceiling(upward(final)),
        proof="Absolute half-FMA recurrence followed by SDK f32 signed reduction and half narrow; globally correlated L1 counted once; gradual underflow retained.",
    )
