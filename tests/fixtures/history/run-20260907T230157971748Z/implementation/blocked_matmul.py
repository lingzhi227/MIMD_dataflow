"""Native-order two-level half partials / float merge semantics."""

import numpy as np
from frontend import check
from binary16 import matmul


def evaluate(a, b, block_size):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    check(
        a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0], "blocked contraction shapes"
    )
    check(
        type(block_size) is int and 1 <= block_size <= a.shape[1],
        "blocked contraction extent",
    )
    total = np.zeros((a.shape[0], b.shape[1]), np.float32)
    for k in range(0, a.shape[1], block_size):
        partial = matmul(a[:, k : k + block_size], b[k : k + block_size])
        total = np.asarray(total + partial.astype(np.float32), np.float32)
    check(
        np.all(np.isfinite(total)) and np.all(np.abs(total) <= 65504),
        "blocked final half range",
    )
    return total.astype(np.float16).astype(float)


def finite_bound(left, right, inner, block_size):
    """Monotone half partial / f32 merge / half narrow magnitude envelope."""
    import math
    from input_contracts import half_dot_bound
    from float32 import f32
    from binary16 import quantize

    check(
        type(inner) is int
        and type(block_size) is int
        and 1 <= block_size <= inner
        and inner % block_size == 0,
        "bounded divisible local blocks",
    )
    partial = half_dot_bound(left, right, block_size)
    total = 0.0
    for _ in range(inner // block_size):
        total = f32(total + partial)
    check(math.isfinite(total) and total <= 65504, "local block merge may overflow")
    return quantize(total)
