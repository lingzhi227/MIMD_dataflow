"""Nonsymmetric row-diagonally-dominant systems and BiCGStab guard cases."""



import numpy as np
from solver_fixtures import batches as cg_batches, check as cg_check, original_apply


def batches(count=10):
    out = cg_batches(count=count)
    for epoch, b in enumerate(out):
        n = len(b["rhs"])
        if epoch in (0, 2, 3, 5, 6):
            diag = np.ones(n)
            locations = {}
            for col in range(n):
                for q in range(b["column_offsets"][col], b["column_offsets"][col + 1]):
                    row = b["row_indices"][q]
                    if row == col:
                        locations[row] = q
                    else:
                        b["values"][q] = float(
                            np.float32(b["values"][q] * (0.5, 1.0, 2.0)[col % 3])
                        )
                        diag[row] += abs(b["values"][q])
            for row, q in locations.items():
                b["values"][q] = float(np.float32(diag[row]))
        if epoch == 4:
            b["values"] = [0.0] * len(b["values"])
        if epoch == 8:
            for col in range(n):
                for q in range(b["column_offsets"][col], b["column_offsets"][col + 1]):
                    b["values"][q] = 2.0 if b["row_indices"][q] == col else 0.0
            b.update(rhs=[1.0] * n, initial=[0.0] * n)
        if epoch == 9:
            entries = {
                (b["row_indices"][q], col): 0.0
                for col in range(n)
                for q in range(b["column_offsets"][col], b["column_offsets"][col + 1])
            }
            for col in range(n):
                entries[col, col] = 1.0 if col % 2 == 0 else 0.0
                key = (col ^ 1, col)
                if key not in entries:
                    victim = next(
                        k
                        for k in entries
                        if k[0] // 128 == col // 128
                        and k[1] // 128 == col // 128
                        and k[0] != k[1]
                        and k[0] != (k[1] ^ 1)
                    )
                    del entries[victim]
                entries[key] = 1.0
            vals = []
            rows = []
            offs = [0]
            for col in range(n):
                for row in sorted(row for row, c in entries if c == col):
                    rows.append(row)
                    vals.append(entries[row, col])
                offs.append(len(rows))
            b.update(
                values=vals,
                row_indices=rows,
                column_offsets=offs,
                rhs=[1.0 if i % 2 == 0 else 0.0 for i in range(n)],
                initial=[0.0] * n,
            )
        b["iteration_limit"][0] = min(b["iteration_limit"][0], 32)
    return out


def check(b, o):
    if not any(b["values"]):
        assert (
            o["reason"] == [3]
            and o["iterations"] == [0]
            and o["solution"] == b["initial"]
        )
        residual = float(
            np.linalg.norm(np.array(b["rhs"]) - original_apply(b, o["solution"]))
        )
        assert residual > 0 and o["true_residual_norm"][0] > 0
        return dict(
            contract="original-CSC-BiCGStab-residual-v1",
            passed=True,
            reason=3,
            iterations=0,
            true_residual=residual,
            strict_requested_tolerance_met=False,
        )
    if o["reason"] == [3] and max(map(abs, b["rhs"])) > 1e-25:
        r = np.asarray(b["rhs"]) - original_apply(b, b["initial"])
        v = original_apply(b, r)
        alpha = float(np.dot(r, r) / np.dot(r, v))
        s = r - alpha * v
        t = original_apply(b, s)
        assert np.dot(t, s) == 0 and np.dot(t, t) > 0
        assert (
            o["iterations"] == [0]
            and o["solution"] == b["initial"]
            and o["true_residual_norm"][0] > 0
        )
        return dict(
            contract="original-CSC-BiCGStab-residual-v1",
            passed=True,
            reason=3,
            iterations=0,
            breakdown="zero_omega",
            strict_requested_tolerance_met=False,
        )
    report = cg_check(b, o)
    report["contract"] = "original-CSC-BiCGStab-residual-v1"
    return report
