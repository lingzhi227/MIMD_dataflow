"""The performance comparator must preserve device bits across output ownership."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "compare_fft_layouts",
    repository_root(__file__) / "experiments/compare_fft_layouts.py",
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class LayoutComparisonTests(unittest.TestCase):
    def test_independent_offset_mapping_preserves_signed_zero_and_subnormals(self):
        n, p, t = 4, 2, 2
        bits = np.arange(2 * n**3, dtype=np.uint32).reshape(n, n, n, 2)
        bits[0, 0, 0] = [0, 0x80000000]
        bits[1, 2, 3] = [0x007FFFFF, 0x80000001]
        for layout in ("input_layout", "transposed_pencils"):
            packed = np.empty((p, p, 2 * n * t * t), dtype=np.uint32)
            for y in range(n):
                for x in range(n):
                    for z in range(n):
                        a, b, c = (y, x, z) if layout == "input_layout" else (x, z, y)
                        offset = 2 * (c * t * t + (a % t) * t + b % t)
                        packed[a // t, b // t, offset : offset + 2] = bits[y, x, z]
            actual = comparison.logical_bits(
                packed.view(np.float32),
                dict(N=n, rows=p, T=t, result_layout=layout),
            )
            np.testing.assert_array_equal(actual, bits)


if __name__ == "__main__":
    unittest.main()
