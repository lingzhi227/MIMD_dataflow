"""Application inputs and independent original-input mathematics; no compiler imports."""

import math
from projected_cache_fixtures import (
    batches as prefix_batches,
    original as prefix_math,
    metric,
)


def batches(b, n, s, f):
    if (b, n, s, f) != (3, 256, 512, 512):
        raise ValueError("Only the measured 3x256x512x512 composed fixture is defined")
    rows = prefix_batches(b, n, s)
    for epoch, row in enumerate(rows):
        for name, t, den, shape in (
            ("wu", 0, 512, (n, f)),
            ("wg", 1, 512, (n, f)),
            ("wd", 2, 4096, (f, n)),
        ):
            row[name] = [
                (((i * 13 + j * 5 + epoch * 7 + t * 11) % 33) - 16) / den
                for i in range(shape[0])
                for j in range(shape[1])
            ]
    row = rows[7]
    for name in ("x", "gamma", "key", "value", "cosine"):
        row[name] = [1.0] * len(row[name])
    row["sine"] = [0.0] * len(row["sine"])
    for name in ("wq", "wk", "wv", "wu", "wg"):
        row[name] = [1 / 32] * len(row[name])
    row["wo"] = [1 / 8] * len(row["wo"])
    row["wd"] = [1 / 256] * len(row["wd"])
    return rows


def original(b, n, s, f, batch):
    result = prefix_math(b, n, s, batch)
    z = result["result"]
    norm = []
    for row in range(b):
        inv = 1 / math.sqrt(
            math.fsum(v * v for v in z[row * n : (row + 1) * n]) / n + 1e-6
        )
        norm.extend(z[row * n + i] * batch["gamma"][i] * inv for i in range(n))

    def multiply(a, w, k, width):
        return [
            math.fsum(a[row * k + i] * w[i * width + j] for i in range(k))
            for row in range(b)
            for j in range(width)
        ]

    up = multiply(norm, batch["wu"], n, f)
    gate = multiply(norm, batch["wg"], n, f)
    activation = []
    for value in gate:
        exp = math.exp(-abs(value))
        activation.append(value / (1 + exp) if value >= 0 else value * exp / (1 + exp))
    hidden = [u * a for u, a in zip(up, activation)]
    delta = multiply(hidden, batch["wd"], f, n)
    result.update(
        ffn_normalized=norm,
        up=up,
        gate=gate,
        activation=activation,
        hidden=hidden,
        ffn_delta=delta,
        final_result=[a + d for a, d in zip(z, delta)],
    )
    return result


def check(b, n, s, f, batch, outputs, observed=None):
    assert set(outputs) == {"result", "new_key", "new_value"}
    ideal = original(b, n, s, f, batch)
    actual = dict(
        observed or {},
        final_result=outputs["result"],
        rotated_key=outputs["new_key"],
        value_projection=outputs["new_value"],
    )
    if observed is not None:
        assert set(actual) == set(ideal), "All eighteen original-input stages required"
    checks = {name: metric(values, ideal[name]) for name, values in actual.items()}
    mass = None
    if observed is not None:
        probability = actual["probability"]
        assert len(probability) == b * s and all(0 <= v <= 1 for v in probability)
        mass = max(
            abs(math.fsum(probability[row * s : (row + 1) * s]) - 1) for row in range(b)
        )
        assert mass <= 0.01, "Probability row mass"
    return dict(
        contract="composed-attention-ffn-half-normwise-v1",
        fixed_accuracy_passed=True,
        all_stage_gates=observed is not None,
        metrics=checks,
        max_probability_mass_error=mass,
        limits=dict(relative_l2=0.02, relative_peak=0.03, row_mass=0.01),
    )
