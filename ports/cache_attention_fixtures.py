"""Independent original-input cache attention fixtures and stdlib arithmetic gate."""

import math


def batches(b, n, s):
    rows = []
    for e in range(8):
        x = [((i * 7 + e * 11) % 33 - 16) / 32 for i in range(b * n)]
        query = [((i * 17 + e * 13) % 65 - 32) / 64 for i in range(b * n)]
        key = [
            ((j * 19 + i * 7 + e * 23) % 65 - 32) / 64
            for j in range(s)
            for i in range(n)
        ]
        value = [
            ((j * 11 + i * 23 + e * 17) % 65 - 32) / 64
            for j in range(s)
            for i in range(n)
        ]
        wo = [
            ((i * 29 + j * 13 + e * 7) % 33 - 16) / 256
            for i in range(n)
            for j in range(n)
        ]
        if e == 0:
            query = [0.0] * len(query)
        if e == 2:
            query = [0.25 + (i % 4) / 32 for i in range(b * n)]
            key = [0.25 + (i % 8) / 64 for i in range(s * n)]
            value = [0.125 + (i % 8) / 64 for i in range(s * n)]
        if e == 3:
            query = [0.25] * len(query)
            key = [
                -(1 + ((j * 7 + i * 3) % 16)) / 64 for j in range(s) for i in range(n)
            ]
        if e == 4:
            features = [0, n // 8 - 1, n // 8, n // 2 - 1, n - 1]
            query = [
                float(i == features[row % 5]) for row in range(b) for i in range(n)
            ]
            key = [
                ((j * 17 + i * 11) % 127 - 63) / 128 for j in range(s) for i in range(n)
            ]
            for row in range(b):
                key[((s - 1 - row * 37) % s) * n + features[row % 5]] = 1.0
            value = [
                ((j // (s // 8)) * 16 + (i // (n // 8)) * 2 + (j + i) % 2 - 64) / 128
                for j in range(s)
                for i in range(n)
            ]
        if e == 5:
            query = [0.5 if (i // n + i % n) % 17 == 0 else 0.0 for i in range(b * n)]
            key = [a if (i // n) % 13 == 0 else 0.0 for i, a in enumerate(key)]
        if e == 6:
            query = [(-1.0 if i % 2 else 1.0) * 0.25 for i in range(b * n)]
            key = [
                ((-1.0 if (i + j) % 2 else 1.0) * 0.25 + (j % 8) / 256)
                for j in range(s)
                for i in range(n)
            ]
        rows.append(dict(x=x, query=query, key=key, value=value, wo=wo))
    return rows


def original(b, n, s, batch):
    x, query, key, value, w = [batch[k] for k in ("x", "query", "key", "value", "wo")]
    score = [
        math.fsum(query[row * n + i] * key[j * n + i] for i in range(n))
        for row in range(b)
        for j in range(s)
    ]
    probability = []
    for row in range(b):
        a = [v / math.sqrt(n) for v in score[row * s : (row + 1) * s]]
        peak = max(a)
        e = [math.exp(v - peak) for v in a]
        total = math.fsum(e)
        probability.extend(v / total for v in e)
    context = [
        math.fsum(probability[row * s + j] * value[j * n + i] for j in range(s))
        for row in range(b)
        for i in range(n)
    ]
    delta = [
        math.fsum(context[row * n + i] * w[i * n + j] for i in range(n))
        for row in range(b)
        for j in range(n)
    ]
    return dict(
        score=score,
        probability=probability,
        context=context,
        delta=delta,
        result=[a + c for a, c in zip(x, delta)],
    )


def check(b, n, s, batch, out, observed=None):
    expected = original(b, n, s, batch)
    assert set(out) == {"result"}
    actual = dict(observed or {}, result=out["result"])
    reports = {}
    assert set(actual) == (set(expected) if observed is not None else {"result"})
    for name, got in actual.items():
        want = expected[name]
        if hasattr(got, "ravel"):
            got = got.ravel().tolist()
        assert len(got) == len(want) and all(math.isfinite(v) for v in got)
        error = [a - c for a, c in zip(got, want)]
        l2 = math.sqrt(math.fsum(v * v for v in error)) / max(
            math.sqrt(math.fsum(v * v for v in want)), 1e-12
        )
        peak = max(map(abs, error)) / max(max(map(abs, want)), 1e-12)
        assert l2 <= 0.02 and peak <= 0.03, (name, l2, peak)
        reports[name] = dict(
            relative_l2=l2, relative_peak=peak, fixed_accuracy_passed=True
        )
    mass_error = None
    if "probability" in actual:
        probability = actual["probability"]
        if hasattr(probability, "ravel"):
            probability = probability.ravel().tolist()
        assert all(0 <= v <= 1 for v in probability)
        mass_error = max(
            abs(math.fsum(probability[row * s : (row + 1) * s]) - 1) for row in range(b)
        )
        assert mass_error <= 0.01
    return dict(
        contract="supplied-cache-attention-half-normwise-v1",
        all_stage_gates=observed is not None,
        fixed_accuracy_passed=True,
        stages=reports,
        relative_l2=max(v["relative_l2"] for v in reports.values()),
        peak_scaled_error=max(v["relative_peak"] for v in reports.values()),
        max_row_mass_error=mass_error,
        limits=dict(relative_l2=0.02, peak_scaled_error=0.03, row_mass=0.01),
    )
