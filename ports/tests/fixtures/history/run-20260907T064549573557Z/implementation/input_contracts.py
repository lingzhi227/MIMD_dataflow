"""Executable input preconditions and bounded half-contraction range propagation."""

import math
from frontend import check


def verify_declarations(module):
    for n in module["nodes"]:
        if "abs_bound" in n:
            b = n["abs_bound"]
            check(
                n["op"] == "input"
                and type(b) in (int, float)
                and math.isfinite(b)
                and b >= 0,
                "finite nonnegative input magnitude contract",
            )


def effective_bound(node, global_bound):
    return min(float(global_bound), float(node.get("abs_bound", global_bound)))


def validate_batch(module, batch):
    # Frozen workloads are checked before native execution and SDK deployment.
    # Standalone callers must uphold the same public input preconditions.
    for n in module["nodes"]:
        if n["op"] == "input" and "abs_bound" in n:
            b = effective_bound(n, module["input_bound"])
            check(
                n["host"] in batch
                and all(math.isfinite(v) and abs(v) <= b for v in batch[n["host"]]),
                "input magnitude contract: " + n["host"],
            )


def half_operand_bound(bound):
    """Largest representable nonnegative half satisfying an absolute bound."""
    import numpy as np

    check(math.isfinite(bound) and 0 <= bound <= 65504, "finite half operand bound")
    q = np.float16(bound)
    if float(q) > bound:
        q = np.nextafter(q, np.float16(-math.inf))
    return float(q)


def half_dot_bound(left, right, length):
    """Monotone positive FMA recurrence bounds every signed bounded dot order."""
    from binary16 import fma

    a, b = half_operand_bound(left), half_operand_bound(right)
    check(type(length) is int and 1 <= length <= 512, "bounded half dot length")
    upper = 0.0
    for _ in range(length):
        check(a * b + upper <= 65504, "half contraction may overflow")
        upper = fma(a, b, upper)
    return upper


def half_product_bound(left, right):
    from binary16 import quantize

    a, b = half_operand_bound(left), half_operand_bound(right)
    check(a * b <= 65504, "half product may overflow")
    return quantize(a * b)
