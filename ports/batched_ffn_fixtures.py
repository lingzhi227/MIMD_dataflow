"""Original-input bounded FFN fixtures and independent standard mathematics."""

import math
import numpy as np


def batches(b, n, f):
    assert (b, n, f) == (5, 256, 512), "bounded application fixture geometry"
    rng = np.random.default_rng(81370)
    cases = []
    for i in range(8):
        x = rng.uniform(-1, 1, (5, 256))
        gamma = rng.uniform(-1, 1, (1, 256))
        wu = rng.uniform(-0.03125, 0.03125, (256, 512))
        wg = rng.uniform(-0.03125, 0.03125, (256, 512))
        wd = rng.uniform(-0.00390625, 0.00390625, (512, 256))
        if i == 0:
            x[:] = 0
        if i == 2:
            x[:] = 1
            gamma[:] = 1
            wu[:] = 0.03125
            wg[:] = 0.03125
            wd[:] = 0.00390625
        if i == 3:
            x[:] = 1
            gamma[:] = 1
            wu[:] = 0.03125
            wg[:] = -0.03125
            wd[:] = 0.00390625
        if i == 4:
            x *= 2**-8
        if i == 5:
            x[:] = 0
            x[:, [0, 31, 32, 127, 255]] = np.array([1, -1, 0.5, -0.5, 2**-12])
            gamma[:] = 1
        if i == 6:
            # Cross-shard cancellation and distinct branch/output identities.
            wu[128:] = -wu[:128]
            wg[:128] *= 0.5
            wd[256:] = -wd[:256]
        if i == 7:
            x[:] = np.where(np.indices(x.shape)[1] % 2, 1, -1)
            gamma[:] = 1
        cases.append(
            {
                k: np.asarray(v, np.float16).astype(float).ravel().tolist()
                for k, v in zip(
                    ("x", "gamma", "wu", "wg", "wd"), (x, gamma, wu, wg, wd)
                )
            }
        )
    return cases


def original(b, n, f, batch, epsilon=1e-6):
    x = np.asarray(batch["x"]).reshape(b, n)
    gamma = np.asarray(batch["gamma"]).reshape(n)
    wu = np.asarray(batch["wu"]).reshape(n, f)
    wg = np.asarray(batch["wg"]).reshape(n, f)
    wd = np.asarray(batch["wd"]).reshape(f, n)
    norm = np.array(
        [
            [
                float(v)
                * float(gamma[j])
                / math.sqrt(math.fsum(float(t) ** 2 for t in row) / n + epsilon)
                for j, v in enumerate(row)
            ]
            for row in x
        ]
    )

    def mm(a, w):
        return np.array(
            [
                [
                    math.fsum(float(v) * float(t) for v, t in zip(row, w[:, j]))
                    for j in range(w.shape[1])
                ]
                for row in a
            ]
        )

    up = mm(norm, wu)
    gate = mm(norm, wg)

    def silu(v):
        e = math.exp(-abs(v))
        return v / (1 + e) if v >= 0 else v * e / (1 + e)

    activation = np.array([[silu(v) for v in row] for row in gate])
    hidden = up * activation
    delta = mm(hidden, wd)
    return dict(
        normalized=norm,
        up=up,
        gate=gate,
        activation=activation,
        hidden=hidden,
        delta=delta,
        result=x + delta,
    )


def check(b, n, f, batch, outputs, observations=None):
    assert set(outputs) == {"result"}
    expected = original(b, n, f, batch)
    actual = dict(result=np.array(outputs["result"]).reshape(b, n))
    if observations is not None:
        assert set(observations) == set(expected) - {"result"}
        actual.update(
            {k: np.array(v).reshape(expected[k].shape) for k, v in observations.items()}
        )
    stages = {}
    for k, value in actual.items():
        error = value - expected[k]
        l2 = float(np.linalg.norm(error) / max(np.linalg.norm(expected[k]), 1e-12))
        peak = float(np.max(np.abs(error)) / max(np.max(np.abs(expected[k])), 1e-12))
        assert np.all(np.isfinite(value)) and l2 <= 0.02 and peak <= 0.03, (k, l2, peak)
        stages[k] = dict(relative_l2=l2, peak_scaled_error=peak)
    return dict(
        contract="batched-feed-forward-half-normwise-v1",
        fixed_accuracy_passed=True,
        all_stage_gates=observations is not None,
        stages=stages,
        relative_l2=max(v["relative_l2"] for v in stages.values()),
        peak_scaled_error=max(v["peak_scaled_error"] for v in stages.values()),
        limits=dict(relative_l2=0.02, peak_scaled_error=0.03),
    )
