"""IEEE binary32 native-order arithmetic for explicit mixed precision.

Products of two binary32 values are exact in binary64. A second rounding can
only change binary32 conversion when the rounded binary64 sum is itself a
binary32 midpoint. Resolve those lanes with platform libm fmaf, including
underflow/overflow boundaries. This avoids treating double-rounding as FMA.
"""

import ctypes, ctypes.util
from functools import lru_cache
import numpy as np
from frontend import check


@lru_cache(None)
def primitive(name, arity):
    library = ctypes.CDLL(ctypes.util.find_library("m") or None)
    function = getattr(library, name)
    function.argtypes = [ctypes.c_float] * arity
    function.restype = ctypes.c_float
    return function


def fma(a, b, c):
    a, b, c = np.broadcast_arrays(*[np.asarray(v, np.float32) for v in (a, b, c)])
    check(all(np.all(np.isfinite(v)) for v in (a, b, c)), "finite f32 FMA operands")
    exact_product = a.astype(float) * b.astype(float)
    wide_sum = exact_product + c.astype(float)
    with np.errstate(over="ignore", invalid="ignore"):
        result = wide_sum.astype(np.float32)
        down = np.nextafter(result, np.float32(-np.inf)).astype(float)
        up = np.nextafter(result, np.float32(np.inf)).astype(float)
        middle = (
            (wide_sum == (result.astype(float) + down) * 0.5)
            | (wide_sum == (result.astype(float) + up) * 0.5)
            | ~np.isfinite(result)
        )
    if np.any(middle):
        result = np.array(result, dtype=np.float32, copy=True)
        fn = primitive("fmaf", 3)
        for index in np.flatnonzero(middle):
            result.flat[index] = fn(
                float(a.flat[index]), float(b.flat[index]), float(c.flat[index])
            )
    check(np.all(np.isfinite(result)), "f32 FMA result overflow")
    return result


def matmul(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    check(a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0], "f32 contraction shapes")
    result = np.zeros((a.shape[0], b.shape[1]), np.float32)
    for k in range(a.shape[1]):
        result = fma(a[:, k, None], b[None, k, :], result)
    return result.astype(float)


def unary(name, a):
    a = np.asarray(a, np.float32)
    fn = primitive(name, 1)
    out = np.array([fn(float(v)) for v in a.ravel()], np.float32).reshape(a.shape)
    check(np.all(np.isfinite(out)), "finite native libm result")
    return out


def softmax(a, scale):
    x = np.asarray(a, np.float32) * np.float32(scale)
    e = unary("expf", np.float32(x - x.max(axis=1)[:, None]))
    total = np.zeros(x.shape[0], np.float32)
    for k in range(x.shape[1]):
        total = np.float32(total + e[:, k])
    return np.float32(e / total[:, None]).astype(float)


def rms(a, gamma, epsilon):
    a = np.asarray(a, np.float32)
    gamma = np.asarray(gamma, np.float32)
    total = np.zeros(a.shape[0], np.float32)
    for k in range(a.shape[1]):
        total = np.float32(total + np.float32(a[:, k] * a[:, k]))
    root = unary(
        "sqrtf",
        np.float32(np.float32(total / np.float32(a.shape[1])) + np.float32(epsilon)),
    )
    inv = np.float32(np.float32(1) / root)
    return np.float32(np.float32(a * gamma) * inv[:, None]).astype(float)
