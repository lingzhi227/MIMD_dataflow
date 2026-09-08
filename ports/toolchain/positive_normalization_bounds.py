"""Exact-rational range analysis for half softmax plus weighted contraction.

Preconditions are explicit: finite half exponents in [0,1], at least one exact
one per global row; stationary half local SUM; SDK f32 reduction then half
narrow; correctly rounded half reciprocal of D in [1,512]. The direct SDK math
probe establishes the exp/reciprocal primitives, not application accuracy.
This analysis is not yet an admission rule for a wider cache application domain.
"""

from fractions import Fraction as F
import math, struct
from frontend import check

U16 = F(1, 2**11)
U32 = F(1, 2**24)
HALF_ERROR = F(1, 2**25)


def upward(value):
    result = float(value)
    if F.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    check(F.from_float(result) >= value, "outward rounded analysis constant")
    return result


def half_ceiling(value):
    check(0 <= value <= 65504, "finite half range certificate")
    word = struct.unpack("<H", struct.pack("<e", upward(value)))[0]
    half = struct.unpack("<e", struct.pack("<H", word))[0]
    if F.from_float(half) < value:
        word += 1
        half = struct.unpack("<e", struct.pack("<H", word))[0]
    check(math.isfinite(half) and F.from_float(half) >= value, "half outward range")
    return half


def geometry(sequence, local, participants):
    check(
        all(type(v) is int for v in (sequence, local, participants))
        and participants in (4, 8, 16)
        and 1 <= local <= 512
        and sequence == local * participants
        and sequence <= 512,
        "positive normalization SDK geometry",
    )


def mass_fraction(sequence, local, participants):
    geometry(sequence, local, participants)
    # q(x)>= (1-u16)*x - half-minsubnormal/2. SDK float sums are
    # normal (nonzero terms lie on the half-minsubnormal integer grid).
    a = (1 - U16) ** (local + 1) * (1 - U32) ** (participants - 1)
    error = (sequence + 1) * HALF_ERROR
    # D>=a*T-error and D>=1. Reciprocal is normal; the final half product
    # adds at most HALF_ERROR per probability, including gradual underflow.
    return (1 + U16) ** 2 * (1 + error) / a + sequence * HALF_ERROR


def probability_mass(sequence, local, participants):
    value = mass_fraction(sequence, local, participants)
    return dict(
        bound=upward(value),
        exact_numerator=value.numerator,
        exact_denominator=value.denominator,
        denominator=[1, sequence],
        scope="Finite upper mass envelope under stated SDK arithmetic; not a normalization-accuracy guarantee",
    )


def weighted_contraction(sequence, local, participants, block, value_bound):
    """Count probability mass once over the entire distributed contraction."""
    geometry(sequence, local, participants)
    check(
        type(block) is int and 1 <= block <= local and local % block == 0,
        "positive weighted contraction divisible blocks",
    )
    check(
        type(value_bound) in (int, float)
        and math.isfinite(value_bound)
        and value_bound >= 0,
        "finite value magnitude",
    )
    mass = mass_fraction(sequence, local, participants)
    v = F(value_bound)
    gh = (1 + U16) ** block
    gl = (1 + U32) ** (local // block - 1)
    gg = (1 + U32) ** (participants - 1)
    # Extra half narrowing factors also conservatively cover the direct
    # single-block fast path, where that narrowing is an identity.
    local_bound = (1 + U16) * gl * gh * (v * mass + local * HALF_ERROR) + HALF_ERROR
    global_bound = (
        (1 + U16) ** 2 * gg * gl * gh * (v * mass + sequence * HALF_ERROR)
        + (1 + U16) * gg * participants * HALF_ERROR
        + HALF_ERROR
    )
    return dict(
        probability_mass=upward(mass),
        local=half_ceiling(local_bound),
        reduced=half_ceiling(global_bound),
        scope="Probability mass counted once, half FMA block error plus float local/SDK merge and half output rounding; range only, no new SDK application qualification",
    )
