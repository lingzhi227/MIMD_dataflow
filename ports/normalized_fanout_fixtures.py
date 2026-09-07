"""Independent normalized multi-projection fixtures, sharing one mathematical RMS."""

import numpy as np
from normalized_matmul_fixtures import batches as single_batches, check as single_check


def batches(m, n, count=3):
    assert count in (2, 3)
    result = []
    rng = np.random.default_rng(210107)
    for epoch, b in enumerate(single_batches(m, n)):
        q = np.asarray(b["q"]).reshape(n, n)
        weights = [q, np.roll(q, 1, axis=0), -0.5 * q][:count]
        if epoch == 5:
            weights = [rng.uniform(-1 / 16, 1 / 16, (n, n)) for _ in range(count)]
        result.append(
            dict(
                x=b["x"],
                w=b["w"],
                **{
                    f"weight{i}": np.asarray(v, np.float16)
                    .astype(float)
                    .ravel()
                    .tolist()
                    for i, v in enumerate(weights)
                },
            )
        )
    return result


def check(m, n, count, b, output):
    assert set(b) == {"x", "w"} | {f"weight{i}" for i in range(count)} and set(
        output
    ) == {f"projection{i}" for i in range(count)}
    branches = [
        single_check(
            m,
            n,
            dict(x=b["x"], w=b["w"], q=b[f"weight{i}"]),
            dict(projected=output[f"projection{i}"]),
        )
        for i in range(count)
    ]
    return dict(
        contract="normalized-fanout-half-normwise-v1",
        fixed_accuracy_passed=all(v["fixed_accuracy_passed"] for v in branches),
        branches=branches,
        per_component_accuracy_not_implied=True,
    )
