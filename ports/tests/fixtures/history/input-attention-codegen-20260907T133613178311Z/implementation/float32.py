import math, struct


def f32(x):
    try:
        v = struct.unpack("f", struct.pack("f", x))[0]
    except (OverflowError, struct.error):
        raise ValueError("float32 overflow")
    if not math.isfinite(v):
        raise ValueError("non-finite float32")
    return v


def close(a, b, rtol=3e-5, atol=3e-6):
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and set(a) == set(b)
            and all(close(a[k], b[k], rtol, atol) for k in a)
        )
    if isinstance(a, list):
        return (
            isinstance(b, list)
            and len(a) == len(b)
            and all(close(x, y, rtol, atol) for x, y in zip(a, b))
        )
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol + rtol * abs(b)
