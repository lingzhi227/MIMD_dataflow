"""Original-domain FFT fixtures: no expected transform is supplied to the device."""



import numpy as np


def batches(n):
    rng = np.random.default_rng(2101)
    random = rng.uniform(-1, 1, (n, n, n, 2)).astype(np.float32)
    impulse = np.zeros_like(random)
    impulse[n // 2 - 1, n // 4 + 1, n - 2] = [1, -0.5]
    zero = np.zeros_like(random)
    constant = np.zeros_like(random)
    constant[:] = [0.25, -0.5]
    # Distinct frequencies on all three axes expose swaps and direction errors.
    y, x, z = np.indices((n, n, n))
    angle = 2 * np.pi * (y + 2 * x + 3 * z) / n
    wave = np.stack((np.cos(angle), np.sin(angle)), axis=-1).astype(np.float32)
    cancellation = np.zeros_like(random)
    cancellation[0, 0, 0] = [1, 0]
    cancellation[-1, -1, -1] = [-1, 0]
    return [
        {"x": v.ravel().tolist()}
        for v in (random, impulse, zero, constant, wave, cancellation)
    ]


def check(n, direction, norm, batch, output):
    """Independent separable direct DFT, not the compiler's FFT oracle."""
    values = np.asarray(batch["x"], np.float32).reshape(n, n, n, 2)
    x = values[..., 0].astype(np.float64) + 1j * values[..., 1].astype(np.float64)
    sign = 1 if direction == "inverse" else -1
    matrix = np.exp(sign * 2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n)
    ref = x
    for axis in (2, 1, 0):
        ref = np.moveaxis(np.moveaxis(ref, axis, -1) @ matrix.T, -1, axis)
    if norm == "ortho":
        ref /= np.sqrt(n**3)
    elif (norm == "backward" and direction == "inverse") or (
        norm == "forward" and direction == "forward"
    ):
        ref /= n**3
    assert set(output) == {"spectrum"}
    raw = np.asarray(output["spectrum"], np.float64)
    assert raw.shape == (2 * n**3,)
    raw = raw.reshape(n, n, n, 2)
    actual = raw[..., 0] + 1j * raw[..., 1]
    assert np.all(np.isfinite(actual))
    error = np.abs(actual - ref)
    rn = float(np.linalg.norm(ref.ravel()))
    en = float(np.linalg.norm(error.ravel()))
    if rn == 0:
        assert np.array_equal(actual, ref)
    else:
        assert en / rn <= 2e-5, "direct DFT relative L2 mismatch"
        assert float(error.max()) <= 3e-5 * float(
            np.max(np.abs(ref))
        ), "direct DFT peak-scaled mismatch"
    return dict(
        contract="fft-f32-normwise-v1",
        fixed_accuracy_passed=True,
        relative_l2_error=en / rn if rn else 0.0,
        max_abs_error=float(error.max()),
        per_component_accuracy_not_implied=True,
        reference="independent separable direct complex128 DFT",
    )
