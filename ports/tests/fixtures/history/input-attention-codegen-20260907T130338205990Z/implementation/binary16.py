"""IEEE binary16 reference, grounded in SDK direct/DSR FMA and transport probes."""

import math, struct

if __package__:
    from .frontend import check
else:
    from frontend import check


def bits(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def quantize(x):
    return struct.unpack("<e", struct.pack("<e", x))[0]


def fma(a, b, c):
    # Half operands have at most11significand bits. Binary64 is ample for the
    # bounded finite matrix profile, followed by exactly one half rounding.
    return quantize(float(a) * float(b) + float(c))


def matmul(a, b):
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    check(a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0], "half matrix shapes")
    check(np.all(np.isfinite(a)) and np.all(np.isfinite(b)), "half finite inputs")
    check(
        np.array_equal(a, a.astype(np.float16).astype(float))
        and np.array_equal(b, b.astype(np.float16).astype(float)),
        "half representable inputs",
    )
    out = np.zeros((a.shape[0], b.shape[1]), dtype=float)
    for k in range(a.shape[1]):
        # A half product is exact in f64; one f16 cast models the fused update.
        out = np.asarray(out + a[:, k, None] * b[None, k, :], dtype=np.float16).astype(
            float
        )
        check(np.all(np.isfinite(out)), "half finite accumulation domain")
    return out


def roundoff(a, b, actual):
    """Conservative finite half dot envelope; exact trajectory is audited separately."""
    import numpy as np

    a, b, actual = [np.asarray(v, dtype=float) for v in (a, b, actual)]
    check(
        a.ndim == b.ndim == 2
        and a.shape[1] == b.shape[0]
        and actual.shape == (a.shape[0], b.shape[1]),
        "half roundoff shapes",
    )
    check(
        all(np.all(np.isfinite(v)) for v in (a, b, actual)),
        "half finite roundoff domain",
    )
    ops = 2 * a.shape[1] + 1
    u = 2**-11
    check(ops * u < 1, "half bounded reduction length")
    gamma = ops * u / (1 - ops * u)
    magnitude = np.abs(a) @ np.abs(b)
    reference = a @ b
    # Include gradual underflow error, not an assumption that subnormals flush.
    bound = np.nextafter(
        (gamma + ops * 2**-53) * magnitude + ops * 2**-24 / (1 - ops * u), np.inf
    )
    error = np.abs(actual - reference)
    check(np.all(error <= bound), "half dot forward error bound exceeded")
    return dict(
        contract="binary16-fma-relaxed-v1",
        forward_bound_passed=True,
        gamma=gamma,
        max_abs_error=float(error.max()),
        max_error_over_bound=float((error / bound).max()),
        native_f32_accuracy_not_implied=True,
    )


def assert_bits_equal(actual, expected, context):
    """Locate the first exact half-word mismatch for human and agent debuggers."""
    import numpy as np

    a, b = np.asarray(actual), np.asarray(expected)
    check(a.shape == b.shape, f"{context}: half word shapes {a.shape} != {b.shape}")
    bad = np.argwhere(a != b)
    if len(bad):
        index = tuple(int(i) for i in bad[0])
        got, want = int(a[index]), int(b[index])
        decode = lambda word: struct.unpack("<e", struct.pack("<H", word))[0]
        raise ValueError(
            f"{context}: first mismatch at {index}, got 0x{got:04x} ({decode(got)}), "
            f"expected 0x{want:04x} ({decode(want)}); {len(bad)} differing words"
        )
