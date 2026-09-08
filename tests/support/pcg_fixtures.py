"""Jacobi-PCG acceptance systems built from original sparse entries."""



import numpy as np
from solver_fixtures import batches as cg_batches, check as cg_check


def batches(count=9):
    out = cg_batches(count=count)
    for epoch, b in enumerate(out):
        n = len(b["rhs"])
        if epoch in (0, 6):
            scale = np.array([2.0 ** ((i % 4) - 2) for i in range(n)])
            for col in range(n):
                for q in range(b["column_offsets"][col], b["column_offsets"][col + 1]):
                    b["values"][q] = float(
                        np.float32(
                            b["values"][q] * scale[col] * scale[b["row_indices"][q]]
                        )
                    )
        if epoch == 4:
            entries = {
                (b["row_indices"][q], col): (1.0 if b["row_indices"][q] == col else 0.0)
                for col in range(n)
                for q in range(b["column_offsets"][col], b["column_offsets"][col + 1])
            }
            for col in range(n):
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
                entries[key] = 2.0
            values = []
            rows = []
            offsets = [0]
            for col in range(n):
                for row in sorted(row for row, c in entries if c == col):
                    rows.append(row)
                    values.append(entries[row, col])
                offsets.append(len(rows))
            b.update(
                values=values,
                row_indices=rows,
                column_offsets=offsets,
                rhs=[1.0 if i % 2 == 0 else -1.0 for i in range(n)],
            )
        if epoch == 8:
            for col in range(n):
                for q in range(b["column_offsets"][col], b["column_offsets"][col + 1]):
                    b["values"][q] = (
                        2.0 ** ((col % 9) - 4) if b["row_indices"][q] == col else 0.0
                    )
            b.update(rhs=[1.0] * n, initial=[0.0] * n, iteration_limit=[64])
    return out


def check(batch, out):
    if out["reason"] == [2]:
        # Controlled positive-diagonal indefinite block matrix has A*b=-b.
        from solver_fixtures import original_apply

        np.testing.assert_array_equal(
            original_apply(batch, batch["rhs"]), -np.array(batch["rhs"])
        )
        assert out["iterations"] == [0] and out["solution"] == batch["initial"]
        return dict(
            contract="original-CSC-PCG-residual-v1",
            passed=True,
            reason=2,
            iterations=0,
            strict_requested_tolerance_met=False,
        )
    report = cg_check(batch, out)
    report["contract"] = "original-CSC-PCG-residual-v1"
    return report
