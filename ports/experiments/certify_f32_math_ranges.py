"""Rational forward-error bounds for pinned SDK normalized math expressions.

Conditional on binary32 round-to-nearest arithmetic (fusion may omit a rounding),
truncating finite float-to-i16 conversion and exact normal power-of-two scaling.
These are source-expression range proofs, not exhaustive target execution.
"""

import json, struct, sys, hashlib
from fractions import Fraction as F
from pathlib import Path

U = F(1, 2**24)
ETA = F(1, 2**126)  # conservatively also covers subnormal flushing


class Value:
    def __init__(self, lo, hi=None, error=F(0)):
        self.lo = F(lo)
        self.hi = F(lo if hi is None else hi)
        self.error = error

    def magnitude(self):
        return max(abs(self.lo), abs(self.hi))

    def add(self, b):
        lo, hi = self.lo + b.lo, self.hi + b.hi
        propagated = self.error + b.error
        return Value(
            lo, hi, propagated + U * (max(abs(lo), abs(hi)) + propagated) + ETA
        )

    def mul(self, b):
        products = [a * c for a in (self.lo, self.hi) for c in (b.lo, b.hi)]
        propagated = (
            self.magnitude() * b.error
            + b.magnitude() * self.error
            + self.error * b.error
        )
        return Value(
            min(products),
            max(products),
            propagated + U * (max(map(abs, products)) + propagated) + ETA,
        )

    def neg(self):
        return Value(-self.hi, -self.lo, self.error)


def sqrt_bounds():
    error = F(37, 1000)
    steps = []
    x = Value(1, 4)
    z = Value(-2, F(-1, 2))  # exact power-of-two multiplication
    for _ in range(2):
        y = Value((1 - error) / 2, 1 + error)
        result = y.add(y.mul(Value(F(1, 2)).add(z.mul(y).mul(y))))
        normalized_rounding = 2 * result.error
        assert normalized_rounding < 32 * U
        exact_error = error * error * (3 + error) / 2
        error = exact_error + normalized_rounding
        steps.append(
            dict(rounding_upper=float(normalized_rounding), error_upper=float(error))
        )
    y = Value((1 - error) / 2, 1 + error)
    a = x.mul(y)
    # Multiplication by .5 is exact in this normal domain.
    half_y = Value(y.lo / 2, y.hi / 2)
    corrected = a.add(half_y.mul(x.add(a.mul(a).neg())))
    assert corrected.error < 128 * U
    final_error = error * error * (3 + error) / 2 + corrected.error
    assert final_error < F(1, 2**16)
    return dict(
        seed_absolute_ratio_error=0.037,
        refinements=steps,
        final_rounding_upper=float(corrected.error),
        relative_error_upper=float(final_error),
        reserved_relative_error=2**-16,
    )


def constant(word):
    return F(float(struct.unpack("<f", struct.pack("<I", word))[0]))


def exp_bounds():
    # For x in [-80,0], n=trunc(fl(x*c)), and |n-fl(x*c)|<1.
    c = constant(0x40B8AA3B)
    left = constant(0x3E317218)
    right = constant(0x3002E300)
    nmax = 465
    residual = 80 * abs(1 - c * left) + left * (1 + 80 * c * U + ETA) + nmax * right
    # Bound both multiplications and additions in r1/r, with no fusion assumed.
    residual += U * (nmax * left + 160 + nmax * right + 1) + 8 * ETA
    assert residual < F(18, 100)
    r = Value(-residual, residual)
    y = Value(constant(0x3D2AB856))
    for word in (0x3E2ABF2B, 0x3EFFFFFF, 0x3F7FFFFF, 0x3F800000):
        y = Value(constant(word)).add(y.mul(r))
    product = y.mul(Value(1, constant(0x3FD744FD)))
    lower = (product.lo - product.error) * F(1, 2**117)
    upper = product.hi + product.error
    assert lower > F(1, 2**126) and upper < 3
    # At exactly zero, n=r=0, the last Horner addition is exactly 1.
    return dict(
        input_interval=[-80, 0],
        reduced_argument_absolute_upper=float(residual),
        result_positive_lower=float(lower),
        result_upper=float(upper),
        zero_image=1,
        source_finite_normal_scaling_exponent_interval=[-117, 0],
    )


if __name__ == "__main__":
    out = Path(sys.argv[1])
    assert not out.exists()
    source = Path(__file__).resolve().parents[1] / "references/sdk-math-2.10.1/math.csl"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == "ef7233d3f2a43be553b512d0579ce284030f448a68b78b7b933814c48c2410c0"
    result = dict(
        passed=True,
        scope=__doc__,
        sqrt=sqrt_bounds(),
        exp=exp_bounds(),
        sdk_math_sha256=digest,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
