"""Independent application inputs and exact witnesses for half matrix kernels."""



import numpy as np
from toolchain.binary16 import roundoff


def batches(rows=64, k=64, cols=64, count=6):
    result = []
    for e in range(count):
        rng = np.random.default_rng(1801 + e)
        a = rng.uniform(-0.5, 0.5, (rows, k)).astype(np.float16)
        b = rng.uniform(-0.5, 0.5, (k, cols)).astype(np.float16)
        if e % 6 == 1:
            a[:] = 0
            a[np.arange(rows), np.arange(rows) % k] = 1
            b = (
                np.arange(k * cols).reshape(k, cols)
                / (2.0 ** int(np.ceil(np.log2(k * cols))))
                - 0.5
            ).astype(np.float16)
        elif e % 6 == 2:
            a[:] = 0
        elif e % 6 == 3:
            a[:] = 0
            b[:] = 0
            a[:, 0] = -(1 + 2**-10)
            a[:, 1] = 1 + 2**-10
            b[0, :] = 1
            b[1, :] = 1 + 2**-10
        elif e % 6 == 4:
            a[:] = 2**-24
            b[:] = 0
            b[np.arange(cols) % k, np.arange(cols)] = 1
        elif e % 6 == 5:
            a[:] = 0
            a[np.arange(rows), np.arange(rows) % k] = -1
        result.append(
            dict(a=a.astype(float).ravel().tolist(), b=b.astype(float).ravel().tolist())
        )
    return result


def check(batch, output, rows=64, k=64, cols=64):
    a = np.asarray(batch["a"]).reshape(rows, k)
    b = np.asarray(batch["b"]).reshape(k, cols)
    got = np.asarray(output["result"]).reshape(rows, cols)
    proof = roundoff(a, b, got)
    witnesses = []
    if not np.any(a):
        np.testing.assert_array_equal(got, 0)
        witnesses.append("exact_zero_reset")
    if np.all(a == 2**-24):
        np.testing.assert_array_equal(got, 2**-24)
        witnesses.append("minimum_half_subnormal_preserved")
    if (
        np.all(a[:, 0] == -(1 + 2**-10))
        and np.all(a[:, 1] == 1 + 2**-10)
        and not np.any(a[:, 2:])
    ):
        np.testing.assert_array_equal(got, 0.00097751617431640625)
        witnesses.append("fused_not_split_half_rounding")
    if (
        np.count_nonzero(a) == rows
        and np.all(np.count_nonzero(a, axis=1) == 1)
        and np.all(np.abs(a[a != 0]) == 1)
    ):
        np.testing.assert_array_equal(got, a @ b)
        witnesses.append("exact_signed_row_selection")
    return dict(**proof, exact_witnesses=witnesses)
