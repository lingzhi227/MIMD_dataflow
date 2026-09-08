"""Half reduction order of Decode's runtime Y routes, checked by SDK witnesses.

This differs from the other grouped-GEMV layout. At an even midpoint root, rd0
receives HEAD and rd1 TAIL; all_reduce consumes rd0 before rd1. Never infer
addition association just from the direction names in a different library.
"""

import numpy as np
from frontend import check


def reduce(values):
    check(len(values) in (2, 4), "Decode admitted midpoint tree sizes")
    q = lambda a: np.asarray(a, np.float16).astype(float)
    root = len(values) // 2
    head = values[0]
    for i in range(1, root):
        head = q(head + values[i])
    result = q(values[root] + head)
    if root + 1 < len(values):
        tail = values[-1]
        for i in range(len(values) - 2, root, -1):
            tail = q(tail + values[i])
        result = q(result + tail)
    return result


def grouped(values, size, root):
    check(root == size // 2, "Decode midpoint root")
    groups = np.asarray(
        [reduce(values[i : i + size]) for i in range(0, len(values), size)]
    )
    return groups, reduce(groups)
