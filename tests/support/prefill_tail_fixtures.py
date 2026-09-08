"""Original-input math for a supplied-attention-output projection and normalized FFN."""



import math
import numpy as np
from feed_forward_fixtures import batches as ff_batches, reference as ff_reference


def batches(m, n, f, p):
    rng = np.random.default_rng(210117)
    rows = []
    for epoch, b in enumerate(ff_batches(m, n, f, p)):
        a = rng.uniform(-0.125, 0.125, (m, n)).astype(np.float16)
        o = rng.uniform(-0.0078125, 0.0078125, (n, n)).astype(np.float16)
        if epoch in (2, 7):
            a.fill(0)
        if epoch == 3:
            a.fill(0.125)
            o.fill(-0.0078125)
        if epoch == 4:
            o.fill(0)
        if epoch == 5:
            a.fill(0.125)
            o.fill(0.0078125)
        if epoch == 6:
            a[:, :] = np.where(np.arange(n) % 2, 0.125, -0.125)
            o.fill(0.0078125)
        item = {k: v for k, v in b.items() if k != "z"}
        item.update(
            attention=a.astype(float).ravel().tolist(),
            output_weight=o.astype(float).ravel().tolist(),
            residual=b["z"],
        )
        rows.append(item)
    return rows


def reference(m, n, f, epsilon, b):
    a = np.asarray(b["attention"]).reshape(m, n)
    o = np.asarray(b["output_weight"]).reshape(n, n)
    projection = np.asarray(
        [
            [math.fsum(float(x) * float(y) for x, y in zip(row, col)) for col in o.T]
            for row in a
        ]
    )
    z = projection + np.asarray(b["residual"]).reshape(m, n)
    ff = {k: b[k] for k in ("gamma", "up_weight", "gate_weight", "down_weight")}
    ff["z"] = z.ravel().tolist()
    final, delta = ff_reference(m, n, f, epsilon, ff)
    return projection, z, delta, final


def check(m, n, f, epsilon, b, out, projection=None, delta=None):
    expected, _, expected_delta, final = reference(m, n, f, epsilon, b)
    assert set(out) == {"output"}

    def measure(actual, wanted):
        actual = np.asarray(actual).reshape(m, n)
        err = actual - wanted
        l2 = float(np.linalg.norm(err)) / max(float(np.linalg.norm(wanted)), 1e-30)
        peak = float(np.max(np.abs(err))) / max(float(np.max(np.abs(wanted))), 1e-30)
        assert np.all(np.isfinite(actual)) and l2 <= 0.02 and peak <= 0.03, (l2, peak)
        return dict(relative_l2=l2, peak_scaled_error=peak, fixed_accuracy_passed=True)

    result = dict(
        measure(out["output"], final),
        contract="supplied-attention-tail-half-normwise-v1",
    )
    if projection is not None:
        result["projection"] = measure(projection, expected)
    if delta is not None:
        result["mlp_delta"] = measure(delta, expected_delta)
    return result
