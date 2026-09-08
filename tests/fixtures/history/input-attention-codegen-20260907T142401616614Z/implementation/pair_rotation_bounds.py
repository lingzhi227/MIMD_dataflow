"""Absolute range for explicit four-product half pair transforms."""

from binary16 import quantize
from frontend import check
from input_contracts import half_operand_bound, half_product_bound


def bound(value, cosine, sine):
    x, c, s = map(half_operand_bound, (value, cosine, sine))
    p0, p1 = half_product_bound(x, c), half_product_bound(x, s)
    check(max(p0 + p1, x * (c + s)) <= 65504, "pair transform may overflow")
    target = quantize(p0 + p1)
    native = quantize(x * (c + s))
    return dict(
        value_absolute=x,
        cosine_absolute=c,
        sine_absolute=s,
        target_absolute=target,
        native_absolute=native,
        output_absolute=max(target, native),
        scope="Absolute range for either explicit pair order; separate half products then add/subtract, plus single-rounded native path. No unit-circle assumption or accuracy claim.",
    )
