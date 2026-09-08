"""F32 FMA recurrence in the shared two-hop block order (not native k order)."""

import numpy as np
from frontend import check
from float32_arithmetic import fma
from mesh_twohop import block_index


def evaluate(a, b, p):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    check(
        a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0],
        "mixed target contraction shapes",
    )
    m, k = a.shape
    n = b.shape[1]
    check(all(v % p == 0 for v in (m, k, n)), "mixed target tile divisibility")
    mt, kt, nt = m // p, k // p, n // p
    out = np.zeros((m, n), np.float32)
    for y in range(p):
        for x in range(p):
            acc = np.zeros((mt, nt), np.float32)
            for step in range(p):
                block = block_index(p, y, x, step)
                for i in range(kt):
                    index = block * kt + i
                    acc = fma(
                        a[y * mt : (y + 1) * mt, index, None],
                        b[None, index, x * nt : (x + 1) * nt],
                        acc,
                    )
            out[y * mt : (y + 1) * mt, x * nt : (x + 1) * nt] = acc
    return out
