"""Conservative probability-mass and resident half-FMA value bounds."""

import functools, hashlib, math, struct
import numpy as np
from frontend import check
from sdk_math_reference import exp_f16_nonpositive


@functools.lru_cache(None)
def exponential_certificate():
    values = []
    for word in range(0x8000, 0xFC00):
        value = struct.unpack("<e", struct.pack("<H", word))[0]
        result = exp_f16_nonpositive(value)
        check(math.isfinite(result) and 0 <= result <= 1, "SDK nonpositive exp range")
        values.append(result)
    check(exp_f16_nonpositive(0.0) == 1, "SDK exp zero anchor")
    return dict(
        operands=len(values) + 1,
        minimum=min(values),
        maximum=max(values),
        zero_image=1,
        table_sha256=hashlib.sha256(
            b"".join(struct.pack("<e", v) for v in values)
        ).hexdigest(),
        scope="Exhaustive finite nonpositive binary16 domain of the pinned SDK arithmetic model; source semantics and executed SDK comparisons bind this model to the target",
    )


def bound(m, p, value_bound):
    check(
        type(m) is int
        and type(p) is int
        and p in (4, 8)
        and 1 <= m <= 512
        and m % p == 0,
        "attention range geometry",
    )
    check(
        math.isfinite(value_bound) and 0 <= value_bound <= 1, "attention bounded values"
    )
    cert = exponential_certificate()
    u = 2**-11
    eta = 2**-25
    # Positive half addends sum exactly in the subnormal interval. Otherwise
    # each addition obeys the normal relative bound. M+P dominates both the
    # native row reduction and the declared distributed local/chain depth.
    depth = m + p
    denominator_upper = m * (1 + u) ** depth
    check(denominator_upper < 2**14, "normal finite reciprocal of positive softmax sum")
    mass = (1 + u) ** 2 / (1 - u) ** depth + m * eta
    growth = (1 + u) ** m
    absolute = (
        0.0
        if value_bound == 0
        else growth * mass * value_bound + eta * (growth - 1) / u
    )
    # Outward half rounding gives a representable internal tensor contract.
    rounded = float(np.float16(absolute))
    if absolute:
        rounded = float(np.nextafter(np.float16(rounded), np.float16(math.inf)))
    return dict(
        value_input_absolute=value_bound,
        probability_mass_upper=mass,
        denominator_upper=denominator_upper,
        value_output_absolute=rounded,
        positive_sum_depth_bound=depth,
        half_unit_roundoff=u,
        half_underflow_absolute=eta,
        exp_certificate=cert,
        assumptions=[
            "finite half logits, positive scale and exact selected row maximum",
            "exp(0)=1, every shifted exp in [0,1]",
            "half local sums and phase-joined row reduction; reciprocal remains normal for this domain",
            "half probability product and M fused half value updates",
            "no masks, dropout, or modified probability normalization",
        ],
        scope="Range bound only; never a substitute for original-input branch accuracy",
    )
