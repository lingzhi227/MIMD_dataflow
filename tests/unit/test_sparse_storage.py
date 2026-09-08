"""Sparse storage contracts checked against original entries and pinned SDK."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import importlib.util
import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from sparse_storage import CSC, Capacity, distribute_x, gather_y, geometry, partition


def matrix(rows, cols, entries):
    entries = sorted(entries, key=lambda e: (e[1], e[0]))
    offsets = [0]
    for col in range(cols):
        offsets.append(offsets[-1] + sum(c == col for _, c, _ in entries))
    return CSC(
        rows, cols, offsets, [r for r, _, _ in entries], [v for _, _, v in entries]
    )


class SparseStorage(unittest.TestCase):
    def test_integer_and_canonical_contracts(self):
        invalid = [
            (3, 1, [0, 1], [True], [1.0]),
            (3, 1, [0, 1], [1.0], [1.0]),
            (3, 1, [0.0, 1], [1], [1.0]),
            (3, 1, [0, 1], [-1], [1.0]),
            (3, 1, [0, 1], [3], [1.0]),
            (3, 2, [0, 2, 1], [0], [1.0]),
            (3, 1, [0, 2], [1, 1], [1.0, 2.0]),
            (3, 1, [0, 2], [2, 0], [1.0, 2.0]),
            (3, 1, [0, 1], [0], [float("nan")]),
            (3, 1, [0, 1], [0], [0.1]),
        ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                CSC(*args)
        # Global indices are genuinely u32, never rounded through float32.
        a = CSC(2**24 + 3, 1, [0, 1], [2**24 + 1], [1.0])
        self.assertEqual(list(a.entries()), [(2**24 + 1, 0, 1.0)])

    def test_empty_and_capacity_boundaries(self):
        a = CSC(7, 9, [0] * 10, [], [])
        p = partition(a, 4, 3)
        self.assertEqual(p["capacity"], dict(nnz=1, columns=1, rows=1))
        self.assertTrue(all(t["local_nnz"] == [0] for row in p["tiles"] for t in row))
        with self.assertRaises(ValueError):
            partition(matrix(8, 8, [(0, 0, 0.0), (1, 0, 2.0)]), 4, 2, Capacity(1, 1, 2))
        with self.assertRaises(ValueError):
            Capacity(65535, 1, 1)
        with self.assertRaises(ValueError):
            partition(a, 4, 3, Capacity(6000, 2000, 1000))
        with self.assertRaises(ValueError):
            geometry(8, 8, 3, 4)
        with self.assertRaises(ValueError):
            geometry(8, 262140, 4, 1)

    def test_original_entries_and_padding(self):
        rng = random.Random(47)
        for m, n, h, w in [(7, 9, 4, 3), (513, 259, 8, 4), (32, 17, 4, 4)]:
            keys = {(rng.randrange(m), rng.randrange(n)) for _ in range(4 * m)}
            # Force last indices and an explicit zero; remove an entire column.
            keys = {key for key in keys if key[1] != 1} | {(m - 1, n - 1)}
            entries = [(r, c, float(rng.randrange(-8, 9))) for r, c in sorted(keys)]
            entries[0] = (*entries[0][:2], -0.0)
            a = matrix(m, n, entries)
            p = partition(a, h, w)
            x = [float(rng.randrange(-4, 5)) for _ in range(n)]
            distributed = distribute_x(x, m, n, h, w)
            g = p["geometry"]
            reconstructed, products = [], [[] for _ in range(m)]
            for y, row in enumerate(p["tiles"]):
                for px, tile in enumerate(row):
                    for k in range(tile["local_nnz_cols"][0]):
                        col = px * g["block_cols"] + tile["mat_col_idx_buf"][k]
                        lo = tile["mat_col_loc_buf"][k]
                        for j in range(lo, lo + tile["mat_col_len_buf"][k]):
                            r = (
                                y * g["block_rows"]
                                + tile["y_rows_init_buf"][tile["mat_rows_buf"][j]]
                            )
                            v = tile["mat_vals_buf"][j]
                            reconstructed.append((r, col, v))
                            ix, rem = divmod(col, g["block_cols"])
                            iy, iz = divmod(rem, g["local_vec_sz"])
                            products[r].append(v * distributed[iy][ix][iz])
            self.assertEqual(sorted(reconstructed), sorted(entries))
            self.assertEqual(
                [math.fsum(v) for v in products],
                [
                    math.fsum(v * x[c] for r, c, v in entries if r == row)
                    for row in range(m)
                ],
            )
            output = [
                [
                    [
                        float(y * g["block_rows"] + px * g["local_out_vec_sz"] + z)
                        for z in range(g["local_out_vec_sz"])
                    ]
                    for px in range(w)
                ]
                for y in range(h)
            ]
            self.assertEqual(gather_y(output, m, n, h, w), list(range(m)))
            for px in range(w):
                for offset in range(g["local_vec_sz"] * h):
                    if offset >= g["block_cols"] or px * g["block_cols"] + offset >= n:
                        self.assertEqual(
                            distributed[offset // g["local_vec_sz"]][px][
                                offset % g["local_vec_sz"]
                            ],
                            1.0,
                        )

    def test_pinned_sdk_packing(self):
        path = (
            ROOT / "third_party/reference_host/sdk_examples/spmv-hypersparse/preprocess.py"
        )
        spec = importlib.util.spec_from_file_location("pinned_sparse_preprocess", path)
        sdk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sdk)
        rng = random.Random(8)
        m, n, h, w = 512, 513, 8, 4
        keys = {(rng.randrange(m), rng.randrange(n)) for _ in range(4096)}
        entries = [(r, c, float(rng.randrange(-8, 9))) for r, c in sorted(keys)]
        a = matrix(m, n, entries)
        csr = sorted(entries)
        ptr = [0]
        for row in range(m):
            ptr.append(ptr[-1] + sum(r == row for r, _, _ in csr))
        original = sdk.preprocess(
            m,
            n,
            len(entries),
            w,
            h,
            np.asarray(ptr),
            np.asarray([c for _, c, _ in csr]),
            np.asarray(a.column_offsets),
            np.asarray(a.row_indices),
            np.asarray(a.values, dtype=np.float32),
        )
        actual = partition(a, h, w)
        for name in actual["extents"]:
            observed = np.asarray([[t[name] for t in row] for row in actual["tiles"]])
            np.testing.assert_array_equal(observed, original[name])


if __name__ == "__main__":
    unittest.main()
