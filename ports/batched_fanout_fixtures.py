"""Original-domain branch-distinct projection fixtures and independent math."""

import math
import numpy as np


def batches(b, n, f, c):
    from batched_rms_fixtures import batches as rms_batches

    result = rms_batches(b, n)
    rng = np.random.default_rng(73519 + b + n + f + c)
    for e, item in enumerate(result):
        for k in range(c):
            q = rng.uniform(-0.125, 0.125, (n, f))
            if e == 1:
                q = (
                    np.resize(np.array([0.125, -0.0625, 0.03125, 0]), (n, f))
                    * (k + 1)
                    / c
                )
            if e == 4:
                q *= 0.03125
            if e == 6:
                q = np.zeros((n, f))
                q[np.arange(f) % n, np.arange(f)] = (-1) ** k * 0.125
            if e == 7:
                q = np.tile(np.linspace(-0.125, 0.125, f), (n, 1)) * (k + 1) / c
            item["weight" + str(k)] = (
                q.astype(np.float16).astype(float).ravel().tolist()
            )
    return result


def check(b, n, f, c, batch, outputs):
    x = np.asarray(batch["x"]).reshape(b, n)
    w = np.asarray(batch["w"])
    norm = np.asarray(
        [
            [
                v
                * w[j]
                / math.sqrt(math.fsum(float(t) * float(t) for t in row) / n + 1e-6)
                for j, v in enumerate(row)
            ]
            for row in x
        ]
    )
    assert set(outputs) == {"branch" + str(k) for k in range(c)}
    checks = []
    for k in range(c):
        q = np.asarray(batch["weight" + str(k)]).reshape(n, f)
        expected = norm @ q
        actual = np.asarray(outputs["branch" + str(k)]).reshape(b, f)
        err = actual - expected
        l2 = float(np.linalg.norm(err) / max(np.linalg.norm(expected), 1e-30))
        peak = float(np.max(np.abs(err)) / max(np.max(np.abs(expected)), 1e-30))
        assert np.all(np.isfinite(actual)) and l2 <= 0.02 and peak <= 0.03, (
            k,
            l2,
            peak,
        )
        checks.append(dict(branch=k, relative_l2=l2, peak_scaled_error=peak))
    return dict(
        contract="batched-normalized-fanout-half-normwise-v1",
        fixed_accuracy_passed=True,
        branches=checks,
        relative_l2=max(v["relative_l2"] for v in checks),
        peak_scaled_error=max(v["peak_scaled_error"] for v in checks),
        limits=dict(relative_l2=0.02, peak_scaled_error=0.03),
    )
