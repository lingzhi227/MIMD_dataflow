"""Range contract for half local squares and dyadically scaled SDK SUM.

This proves finite intermediate statistics, not an RMS accuracy guarantee.
The caller owns serialized SDK planes and enforces the returned resource lease.
"""

from fractions import Fraction
from frontend import check
from input_contracts import half_operand_bound, half_product_bound, half_dot_bound
from rms_l1_bounds import upward, half_ceiling


def plan_mean_statistic(value, local_features, participants, divisor, capacity):
    for name, number in (
        ("local_features", local_features),
        ("participants", participants),
        ("divisor", divisor),
        ("capacity", capacity),
    ):
        check(type(number) is int and number > 0, "positive integer " + name)
    check(
        local_features <= 512 and participants in (4, 8, 16), "mean statistic geometry"
    )
    check(divisor <= 32768 and divisor & (divisor - 1) == 0, "dyadic u16 divisor")
    check(capacity <= 32767, "mean statistic capacity")
    operand = half_operand_bound(value)
    square = half_product_bound(operand, operand)
    local = half_dot_bound(square, 1.0, local_features)
    # Every finite half and power-of-two scale is exactly representable in f32:
    # smallest nonzero send is 2^-24 / 2^15 = 2^-39, above f32 normal minimum.
    send = (
        Fraction(local, divisor)
        if isinstance(local, int)
        else Fraction(local) / divisor
    )
    upper = send * participants * (1 + Fraction(1, 1 << 24)) ** (participants - 1)
    mean = half_ceiling(upward(upper))
    return dict(
        local_square_bound=square,
        local_sum_bound=local,
        scaled_send_bound=upward(send),
        reduced_f32_bound=upward(upper),
        narrowed_mean_bound=mean,
        standard_rms=divisor == participants * local_features,
        private_buffer_bytes=8 * capacity,
        callback_task=12,
        added_colors=0,
        added_queues=0,
        serialization="borrow caller SDK SUM/broadcast planes until callback; do not overlap SUM/MAX/mean",
        proof="Half product and monotone local addition recurrence must remain finite; exact dyadic half-to-f32 scaling, conservative P-1 f32 addition envelope, outward half ceiling.",
        limitation="Range only. Final half statistic quantization and epsilon-sensitive inverse require an independent accuracy gate. No global memory-fit or performance claim.",
    )


def normalized_mean_bounds(value, gamma, local_features, participants, epsilon):
    """Correlated element/L1 range, with final-half mean underflow retained.

    The result is a candidate lowering certificate. Accuracy remains separately
    checked against original-input mathematics; SDK reciprocal/sqrt are pinned.
    """
    import math
    import struct
    from binary16 import bits
    from sdk_math_reference import rms_inverse_f16

    n = local_features * participants
    plan = plan_mean_statistic(value, local_features, participants, n, 1)
    v, g = Fraction(half_operand_bound(value)), Fraction(half_operand_bound(gamma))
    half_product_bound(float(v), float(g))
    check(
        math.isfinite(epsilon) and 0 < half_operand_bound(epsilon) <= 1,
        "mean RMS epsilon",
    )
    u, u32, eta = Fraction(1, 2048), Fraction(1, 16777216), Fraction(1, 16777216)
    a = (1 - u) ** (local_features + 2) * (1 - u32) ** (participants - 1)
    loss = n * eta / n + eta / 2
    largest_l1 = largest_element = 0.0
    witness = 0.0

    def sqrt_up(q):
        f = math.sqrt(upward(q))
        while Fraction.from_float(f) ** 2 < q:
            f = math.nextafter(f, math.inf)
        return Fraction.from_float(f)

    for word in range(bits(plan["narrowed_mean_bound"]) + 1):
        mean = struct.unpack("<e", struct.pack("<H", word))[0]
        inv = Fraction(rms_inverse_f16(mean, 1, epsilon))
        energy = min(n * v * v, n * (Fraction(mean) + loss) / a)
        root = sqrt_up(n * energy)
        tail = (1 + u) * eta / 2 * inv + eta / 2
        l1 = upward((1 + u) ** 2 * g * inv * root + n * tail)
        element = upward((1 + u) ** 2 * g * inv * min(v, sqrt_up(energy)) + tail)
        if l1 > largest_l1:
            largest_l1, witness = l1, mean
        largest_element = max(largest_element, element)
    check(largest_element <= 65504, "mean normalized result finite")
    return dict(
        l1_bound=largest_l1,
        elementwise_bound=half_ceiling(largest_element),
        mean_witness=witness,
        lower_factor_exact=[str(a.numerator), str(a.denominator)],
        absolute_mean_loss_exact=[str(loss.numerator), str(loss.denominator)],
        statistic_plan=plan,
        proof="M >= (1-u16)^(Nt+2)*(1-u32)^(P-1)*T/N - (eta16+eta16/2). Exact dyadic prescaling. Enumerate half M; T<=N*(M+loss)/A; Cauchy-Schwarz and pinned SDK inverse. Gamma-first/final relative and gradual-underflow rounding retained.",
        scope="Finite correlated range only, not a numerical accuracy guarantee or HLS admission.",
    )
