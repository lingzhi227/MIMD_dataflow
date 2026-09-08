"""Validation-only association for the probed root-zero SDK f32 SUM chain."""

import numpy as np


def half_sum(a, axis):
    v = np.moveaxis(a, axis, 0)
    total = np.asarray(v[-1], np.float32)
    for x in v[-2::-1]:
        total = np.asarray(total + np.asarray(x, np.float32), np.float32)
    return np.asarray(total, np.float16).astype(float)
