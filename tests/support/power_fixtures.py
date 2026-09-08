"""Fixed-step power acceptance; completion is not an eigenpair certificate."""



import math
import numpy as np
from solver_fixtures import batches as cg_batches, original_apply


def batches(count=8):
    out = []
    for epoch, b in enumerate(cg_batches(count=count)):
        n = len(b["initial"])
        initial = np.ones(n, dtype=np.float32)
        steps = 4
        if epoch in (0, 4, 5, 6):
            for col in range(n):
                for q in range(b["column_offsets"][col], b["column_offsets"][col + 1]):
                    value = 1.0
                    if col == 137 and epoch == 0:
                        value = 2.0
                    if col == 137 and epoch in (4, 5):
                        value = -2.0
                    b["values"][q] = value if b["row_indices"][q] == col else 0.0
        if epoch == 0:
            steps = 16
        if epoch == 1:
            b["values"] = [0.0] * len(b["values"])
        if epoch == 2:
            initial[:] = 0
            steps = 3
        if epoch == 3:
            initial = np.asarray([(i % 7 - 3) / 4 for i in range(n)], dtype=np.float32)
            steps = 0
        if epoch in (4, 5):
            initial[:] = 0
            initial[137] = 2 if epoch == 4 else 4
            steps = 3 if epoch == 4 else 4
        if epoch == 6:
            initial[:] = np.float32(1e-30)
            steps = 2
        if epoch == 7:
            # The source matrix for this final warm call is a nontrivial sparse SPD graph.
            b = cg_batches(count=1)[0]
            initial = np.asarray([(i % 9 - 4) / 4 for i in range(n)], dtype=np.float32)
        out.append(
            dict(
                values=b["values"],
                row_indices=b["row_indices"],
                column_offsets=b["column_offsets"],
                initial=initial.tolist(),
                steps=[steps],
            )
        )
    return out


def check(b, o):
    steps = b["steps"][0]
    x = np.asarray(b["initial"], dtype=float)
    reason = 0
    k = 0
    norms = []
    for i in range(steps):
        y = original_apply(b, x)
        nr = float(np.linalg.norm(y))
        norms.append(nr)
        if nr == 0:
            reason = 1
            break
        x = y / nr
        k = i + 1
    assert o["reason"] == [reason] and o["iterations"] == [k]
    got = np.asarray(o["vector"])
    np.testing.assert_allclose(got, x, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(o["norms"][: len(norms)], norms, rtol=3e-5, atol=0)
    assert not any(o["norms"][len(norms) :])
    if k == 0:
        np.testing.assert_array_equal(got, b["initial"])
    else:
        np.testing.assert_allclose(np.linalg.norm(got), 1.0, rtol=3e-6, atol=0)
    if 0 < max(map(abs, b["initial"])) < 1e-25:
        assert o["norms"][0] > 0 and np.any(got)
    denom = float(np.dot(got, got))
    ax = original_apply(b, got)
    rayleigh = float(np.dot(got, ax) / denom) if denom else None
    residual = float(np.linalg.norm(ax - rayleigh * got)) if denom else None
    return dict(
        contract="original-CSC-fixed-power-v1",
        passed=True,
        reason=reason,
        completed_steps=k,
        rayleigh_quotient=rayleigh,
        eigen_residual=residual,
        dominance_certified=False,
    )
