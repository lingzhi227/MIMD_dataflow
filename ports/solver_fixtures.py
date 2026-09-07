"""Independent sparse SPD/termination cases; never imported by lowering."""

import math
import numpy as np


def batches(n=512, count=8):
    result = []
    for epoch in range(count):
        rng = np.random.default_rng(6201 + epoch)
        edges = set()
        while len(edges) < 4 * n:
            i, j = map(int, rng.integers(0, n, 2))
            if i != j:
                edges.add(tuple(sorted((i, j))))
        entries = {(i, i): 0.0 for i in range(n)}
        diagonal = np.ones(n, np.float32)
        for i, j in sorted(edges):
            weight = np.float32(rng.integers(1, 5) / 16)
            entries[i, j] = entries[j, i] = -float(weight)
            diagonal[i] += weight
            diagonal[j] += weight
        for i in range(n):
            entries[i, i] = float(diagonal[i])
        rhs = rng.uniform(-1, 1, n).astype(np.float32)
        initial = np.zeros(n, np.float32)
        limit = 64
        if epoch == 1:
            rhs[:] = 0
        elif epoch == 2:
            rhs[:] = 1
            initial[:] = 1
        elif epoch == 3:
            limit = 1
        elif epoch == 4:
            for key in entries:
                entries[key] = -1.0 if key[0] == key[1] else 0.0
            rhs[:] = 1
        elif epoch == 5:
            limit = 0
        elif epoch == 6:
            initial = rng.uniform(-0.25, 0.25, n).astype(np.float32)
        elif epoch == 7:
            for key in entries:
                entries[key] = 1.0 if key[0] == key[1] else 0.0
            rhs[:] = np.float32(1e-30)
        offsets = [0]
        rows = []
        values = []
        for col in range(n):
            for row in sorted(i for i, j in entries if j == col):
                rows.append(row)
                values.append(entries[row, col])
            offsets.append(len(rows))
        assert len(rows) == 9 * n
        result.append(
            dict(
                values=values,
                row_indices=rows,
                column_offsets=offsets,
                rhs=rhs.tolist(),
                initial=initial.tolist(),
                iteration_limit=[limit],
                tolerances=[float(np.float32(1e-4)), 0.0],
            )
        )
    return result


def original_apply(batch, x):
    terms = [[] for _ in x]
    for col in range(len(x)):
        for p in range(batch["column_offsets"][col], batch["column_offsets"][col + 1]):
            terms[batch["row_indices"][p]].append(
                float(batch["values"][p]) * float(x[col])
            )
    return np.asarray([math.fsum(row) for row in terms])


def check(batch, output):
    x = np.asarray(output["solution"])
    b = np.asarray(batch["rhs"])
    reason = int(output["reason"][0])
    k = int(output["iterations"][0])
    assert reason in (0, 1, 2, 3, 4) and 0 <= k <= batch["iteration_limit"][0]
    true = float(np.linalg.norm(b - original_apply(batch, x)))
    threshold = max(
        batch["tolerances"][0] * float(np.linalg.norm(b)), batch["tolerances"][1]
    )
    if reason == 0:
        # Original double-accumulated operator is separate from device f32 SpMV.
        assert true <= threshold * 1.05 + 1e-7 * float(np.linalg.norm(b)), (
            true,
            threshold,
        )
    expected_reason = 0
    if batch["iteration_limit"][0] in (0, 1):
        expected_reason = 1
    elif all(
        batch["values"][p] == (-1.0 if batch["row_indices"][p] == col else 0.0)
        for col in range(len(x))
        for p in range(batch["column_offsets"][col], batch["column_offsets"][col + 1])
    ):
        expected_reason = 2
    elif 0 < float(np.max(np.abs(b))) < 1e-25:
        expected_reason = 3
    assert reason == expected_reason, (reason, expected_reason, true, threshold)
    if not np.any(b):
        assert k == 0 and np.array_equal(x, np.zeros_like(x))
    if np.array_equal(b, original_apply(batch, batch["initial"])):
        assert k == 0 and np.array_equal(x, batch["initial"])
    return dict(
        contract="original-CSC-CG-residual-v1",
        passed=True,
        reason=reason,
        iterations=k,
        true_residual=true,
        requested_threshold=threshold,
        strict_requested_tolerance_met=(true <= threshold),
        rounding_allowance=0.05 * threshold + 1e-7 * float(np.linalg.norm(b)),
    )
