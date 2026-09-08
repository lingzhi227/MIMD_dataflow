"""Original-input whole-graph fixtures and independent standard-library mathematics."""

import math


def batches(b, n, s):
    from cache_attention_fixtures import batches as cache_batches

    rows = []
    for e, tail in enumerate(cache_batches(b, n, s)):
        x = tail["x"]
        gamma = [((i * 7 + e * 3) % 17) / 16 for i in range(n)]
        weights = [
            [
                (((i * 11 + j * 7 + e * 3 + t * 13) % 33) - 16) / 512
                for i in range(n)
                for j in range(n)
            ]
            for t in range(3)
        ]
        # Coefficients are exactly representable and independently bounded; a unit
        # circle is not assumed by the operator or its range proof.
        cosine = [((i * 3 + e) % 9) / 8 for i in range(n // 2)]
        sine = [((i * 5 + e * 2) % 17 - 8) / 8 for i in range(n // 2)]
        if e == 0:
            gamma = [0.0] * n
        if e == 2:
            x = [0.25 + (i % 3) / 64 for i in range(b * n)]
            gamma = [1.0] * n
            weights = [
                [sign * (1 + (j % 4)) / 256 for i in range(n) for j in range(n)]
                for sign in (1, 1, -1)
            ]
            cosine = [0.75] * (n // 2)
            sine = [0.5] * (n // 2)
        if e == 3:
            x = [0.25] * (b * n)
            gamma = [1.0] * n
            weights = [[float(t + 1) / 256] * (n * n) for t in range(3)]
            cosine = [1.0] * (n // 2)
            sine = [0.0] * (n // 2)
        if e == 4:
            features = [0, n // 8 - 1, n - 1]
            x = [float(i == features[row % 3]) for row in range(b) for i in range(n)]
            gamma = [1.0] * n
            weights = [
                [
                    float(j == ((i + t * (n // 8 + 1)) % n)) / 32
                    for i in range(n)
                    for j in range(n)
                ]
                for t in range(3)
            ]
            cosine = [1.0] * (n // 2)
            sine = [0.0] * (n // 2)
        if e == 5:
            x = [v if i % 17 == 0 else 0.0 for i, v in enumerate(x)]
        if e == 6:
            x = [(-0.25 if i % 2 else 0.25) + (i // n) / 128 for i in range(b * n)]
            gamma = [1.0 if i % 3 else 0.5 for i in range(n)]
            cosine = [0.5] * (n // 2)
            sine = [0.5] * (n // 2)
        rows.append(
            dict(
                x=x,
                gamma=gamma,
                wq=weights[0],
                wk=weights[1],
                wv=weights[2],
                cosine=cosine,
                sine=sine,
                key=tail["key"],
                value=tail["value"],
                wo=tail["wo"],
            )
        )
    return rows


def original(b, n, s, batch):
    x = batch["x"]
    g = batch["gamma"]
    norm = []
    for row in range(b):
        inverse = 1 / math.sqrt(
            math.fsum(x[row * n + i] ** 2 for i in range(n)) / n + 1e-6
        )
        norm.extend(x[row * n + i] * g[i] * inverse for i in range(n))

    def product(w):
        return [
            math.fsum(norm[row * n + i] * w[i * n + j] for i in range(n))
            for row in range(b)
            for j in range(n)
        ]

    q, k, v = [product(batch[name]) for name in ("wq", "wk", "wv")]

    def rotate(a):
        out = []
        for row in range(b):
            for j in range(n // 2):
                even, odd = a[row * n + 2 * j : row * n + 2 * j + 2]
                c, ss = batch["cosine"][j], batch["sine"][j]
                out.extend(
                    (math.fsum((odd * c, -even * ss)), math.fsum((even * c, odd * ss)))
                )
        return out

    rq, rk = rotate(q), rotate(k)
    old_key, old_value, wo = batch["key"], batch["value"], batch["wo"]
    score = [
        math.fsum(rq[row * n + i] * old_key[j * n + i] for i in range(n))
        for row in range(b)
        for j in range(s)
    ]
    probability = []
    for row in range(b):
        scaled = [v / math.sqrt(n) for v in score[row * s : (row + 1) * s]]
        peak = max(scaled)
        exponents = [math.exp(v - peak) for v in scaled]
        total = math.fsum(exponents)
        probability.extend(v / total for v in exponents)
    context = [
        math.fsum(probability[row * s + j] * old_value[j * n + i] for j in range(s))
        for row in range(b)
        for i in range(n)
    ]
    delta = [
        math.fsum(context[row * n + i] * wo[i * n + j] for i in range(n))
        for row in range(b)
        for j in range(n)
    ]
    tail = dict(
        score=score,
        probability=probability,
        context=context,
        delta=delta,
        result=[a + d for a, d in zip(x, delta)],
    )
    return dict(
        normalized=norm,
        query=q,
        key_projection=k,
        value_projection=v,
        rotated_query=rq,
        rotated_key=rk,
        **tail
    )


def metric(actual, expected):
    a = list(actual)
    e = list(expected)
    assert len(a) == len(e) and all(math.isfinite(v) for v in a)
    errors = [x - y for x, y in zip(a, e)]
    norm = math.sqrt(math.fsum(v * v for v in e))
    peak = max(abs(v) for v in e)
    l2 = math.sqrt(math.fsum(v * v for v in errors)) / max(norm, 1e-30)
    maximum = max(abs(v) for v in errors) / max(peak, 1e-30)
    assert l2 <= 0.02 and maximum <= 0.03, (
        "projected cache original-input stage gate",
        l2,
        maximum,
    )
    return dict(relative_l2=l2, relative_peak=maximum)


def check(b, n, s, batch, outputs, observed):
    assert set(outputs) == {"result", "new_key", "new_value"}
    expected = original(b, n, s, batch)
    actual = dict(
        observed,
        result=outputs["result"],
        rotated_key=outputs["new_key"],
        value_projection=outputs["new_value"],
    )
    assert set(actual) == set(expected), "all eleven stages required"
    metrics = {}
    for k, v in expected.items():
        try:
            metrics[k] = metric(actual[k], v)
        except AssertionError as error:
            raise AssertionError((k, *error.args)) from error
    prob = list(actual["probability"])
    assert len(prob) == b * s and all(0 <= v <= 1 for v in prob)
    mass = [abs(math.fsum(prob[row * s : (row + 1) * s]) - 1) for row in range(b)]
    assert max(mass) <= 0.01, "probability row-mass gate"
    return dict(
        contract="projected-cache-half-normwise-v1",
        fixed_accuracy_passed=True,
        all_stage_gates=True,
        metrics=metrics,
        max_probability_mass_error=max(mass),
        limits=dict(relative_l2=0.02, relative_peak=0.03, row_mass=0.01),
    )
