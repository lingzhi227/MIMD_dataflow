"""Eight original-domain fixtures for source-backed batched RMS."""

import numpy as np
from rms_fixtures import batches as rms_batches, check


def batches(b, n):
    result = rms_batches(b, n)
    x = np.zeros((b, n), np.float16)
    for row in range(b):
        x[row, (row * 67 + n - 1) % n] = np.float16((-1) ** row)
    w = np.linspace(-1, 1, n).astype(np.float16)
    result.append(dict(x=x.astype(float).ravel().tolist(), w=w.astype(float).tolist()))
    x = np.tile(np.resize(np.array([-1, 0.5, 0.125, -0.25], np.float16), n), (b, 1))
    for row in range(b):
        x[row] *= np.float16((row + 1) / b)
    result.append(dict(x=x.astype(float).ravel().tolist(), w=np.ones(n).tolist()))
    return result
