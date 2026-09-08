"""Independent branch weights and detection of a single corrupted output."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import sys, unittest
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT))
from normalized_fanout_fixtures import batches, check


class FanoutFixtures(unittest.TestCase):
    def test_final_warm_weights_are_independent(self):
        b = batches(4, 8, 3)[-1]
        q, k, v = [np.asarray(b[f"weight{i}"]).reshape(8, 8) for i in range(3)]
        self.assertFalse(np.array_equal(k, np.roll(q, 1, axis=0)))
        self.assertFalse(np.array_equal(v, -0.5 * q))
        self.assertFalse(np.array_equal(q, k))

    def test_each_output_is_checked(self):
        b = batches(4, 8, 3)[0]
        x = np.asarray(b["x"]).reshape(4, 8)
        w = np.asarray(b["w"])
        norm = x * w / np.sqrt(np.mean(x * x, axis=1)[:, None] + 1e-6)
        outputs = {
            f"projection{i}": (norm @ np.asarray(b[f"weight{i}"]).reshape(8, 8))
            .ravel()
            .tolist()
            for i in range(3)
        }
        self.assertTrue(check(4, 8, 3, b, outputs)["fixed_accuracy_passed"])
        outputs["projection2"][0] += 1
        with self.assertRaises(AssertionError):
            check(4, 8, 3, b, outputs)
