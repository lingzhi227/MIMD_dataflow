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
