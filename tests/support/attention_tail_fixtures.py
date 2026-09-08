"""Original-input supplied Q/K/V attention and projection/FFN tail mathematics."""



import math
import numpy as np
from prefill_tail_fixtures import batches as tail_batches, reference as tail_reference


def batches(m, n, f, p):
    rng = np.random.default_rng(210123)
    result = []
    for epoch, b in enumerate(tail_batches(m, n, f, p)):
        q, k, v = [
            rng.uniform(-0.125, 0.125, (m, n)).astype(np.float16) for _ in range(3)
        ]
        if epoch in (2, 7):
            v.fill(0)
        if epoch == 3:
            q.fill(0.125)
            k.fill(-0.125)
        if epoch == 5:
            v.fill(0.125)
        if epoch == 6:
            q.fill(0)
            k.fill(0)
            v[:] = np.where(np.arange(m)[:, None] % 2, 0.125, -0.125)
        row = {key: value for key, value in b.items() if key != "attention"}
        row.update(
            {
                key: value.astype(float).ravel().tolist()
                for key, value in zip(("q", "k", "v"), (q, k, v))
            }
        )
        result.append(row)
    return result


def contraction(a, b):
    return np.asarray(
        [
            [math.fsum(float(x) * float(y) for x, y in zip(row, col)) for col in b.T]
            for row in a
        ]
    )


def reference(m, n, f, epsilon, scale, b):
    q, k, v = [np.asarray(b[key]).reshape(m, n) for key in ("q", "k", "v")]
    logits = contraction(q, k.T)
    probability = []
    for row in logits:
        scaled = [float(x) * scale for x in row]
        peak = max(scaled)
        exps = [math.exp(x - peak) for x in scaled]
        total = math.fsum(exps)
        probability.append([x / total for x in exps])
    probability = np.asarray(probability)
    attention = contraction(probability, v)
    child = {key: value for key, value in b.items() if key not in ("q", "k", "v")}
    child["attention"] = attention.ravel().tolist()
    projection, z, delta, final = tail_reference(m, n, f, epsilon, child)
    return logits, probability, attention, projection, z, delta, final


def check(
    m,
    n,
    f,
    epsilon,
    scale,
    b,
    out,
    attention=None,
    projection=None,
    delta=None,
    *,
    score=None,
    probability=None
):
    ss, pr, aa, pp, _, dd, yy = reference(m, n, f, epsilon, scale, b)
    assert set(out) == {"output"}

    def measure(actual, wanted, peak_limit=0.03, l2_limit=0.02):
        actual = np.asarray(actual).reshape(wanted.shape)
        error = actual - wanted
        l2 = float(np.linalg.norm(error)) / max(float(np.linalg.norm(wanted)), 1e-30)
        peak = float(np.max(np.abs(error))) / max(float(np.max(np.abs(wanted))), 1e-30)
        assert np.all(np.isfinite(actual)) and l2 <= l2_limit and peak <= peak_limit, (
            l2,
            peak,
            peak_limit,
        )
        return dict(relative_l2=l2, peak_scaled_error=peak, fixed_accuracy_passed=True)

    r = dict(
        measure(out["output"], yy),
        contract="supplied-qkv-attention-tail-half-normwise-v1",
    )
    for key, actual, wanted, limit in [
        ("attention", attention, aa, 0.025),
        ("projection", projection, pp, 0.03),
        ("mlp_delta", delta, dd, 0.03),
    ]:
        if actual is not None:
            r[key] = measure(actual, wanted, limit)
    if score is not None:
        r["score"] = measure(score, ss, 0.02, 0.015)
    if probability is not None:
        value = measure(probability, pr, 0.02, 0.015)
        mass = float(
            np.max(np.abs(np.asarray(probability).reshape(m, m).sum(axis=1) - 1))
        )
        assert mass <= 0.01 and np.all(np.asarray(probability) >= 0), mass
        r["probability"] = dict(value, max_row_mass_error=mass)
    return r
