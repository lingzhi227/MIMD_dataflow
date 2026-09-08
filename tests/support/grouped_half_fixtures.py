"""Row-vector workloads with group-boundary cancellation and distributed ownership."""



import numpy as np
from half_fixtures import batches as base_batches, check as base_check


def batches(k, n, p, groups, count=8):
    result = base_batches(1, k, n, min(count, 6))
    if count > 6:
        a = np.zeros((1, k))
        b = np.zeros((k, n))
        mt = k // p
        values = [-1.0, 2**-11, 1.0, 2**-12, -0.5, 2**-12, 0.5, -(2**-11)]
        for y in range(p):
            a[0, y * mt] = 1
            b[y * mt, :] = values[y]
        result.append(dict(a=a.ravel().tolist(), b=b.ravel().tolist()))
    if count > 7:
        rng = np.random.default_rng(20260906)
        a = rng.uniform(-0.5, 0.5, (1, k)).astype(np.float16).astype(float)
        b = np.zeros((k, n))
        for j in range(n):
            b[(j + 3) % k, j] = 1 if j % 2 else -1
        result.append(dict(a=a.ravel().tolist(), b=b.ravel().tolist()))
    if count > 8:
        raise ValueError("grouped fixture count at most8")
    return result


def check(batch, output, k, n, p, groups):
    proof = base_check(batch, output, 1, k, n)
    a = np.asarray(batch["a"]).reshape(1, k)
    b = np.asarray(batch["b"]).reshape(k, n)
    if np.all(np.count_nonzero(b, axis=0) == 1) and np.all(np.abs(b[b != 0]) == 1):
        expected = np.asarray(
            [
                a[0, np.flatnonzero(b[:, j])[0]] * b[np.flatnonzero(b[:, j])[0], j]
                for j in range(n)
            ]
        )
        np.testing.assert_array_equal(output["result"], expected)
        proof["exact_witnesses"].append("distributed_signed_column_selection")
    return proof
