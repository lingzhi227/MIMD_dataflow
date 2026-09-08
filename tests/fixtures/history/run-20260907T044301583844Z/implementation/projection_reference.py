"""Target-order rectangular contraction witness model, independent of code emission."""

import numpy as np
from mesh_twohop import block_index
from mesh_common import unpack_tiles


def project(s, a, b):
    p, mt, kt, nt = [s[k] for k in ("P", "Mt", "Kt", "Nt")]
    h = np.zeros((p, p, p, mt * nt))
    left = np.zeros((p, p, mt * kt))
    right = np.zeros((p, p, kt * nt))
    for y in range(p):
        for x in range(p):
            acc = np.zeros((mt, nt))
            for r in range(p):
                k = block_index(p, y, x, r)
                aa = a[y * mt : (y + 1) * mt, k * kt : (k + 1) * kt]
                bb = b[k * kt : (k + 1) * kt, x * nt : (x + 1) * nt]
                if r == 0:
                    left[y, x] = aa.ravel(order="F")
                    right[y, x] = bb.ravel(order="C")
                for j in range(kt):
                    acc = (
                        (acc + aa[:, j, None] * bb[None, j, :])
                        .astype(np.float16)
                        .astype(float)
                    )
                h[y, x, r] = acc.ravel(order="F")
    return unpack_tiles(h[:, :, -1], mt, nt, "F"), h.reshape(p, p, -1), left, right
