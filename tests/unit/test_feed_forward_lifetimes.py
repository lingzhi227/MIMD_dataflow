"""FFN live-Z and RMS/projection resource transitions are explicit contracts."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import json, sys, unittest
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from feed_forward_lifetimes import plan
from region_lifetimes import verify
from frontend import Error


class FeedForwardLifetimes(unittest.TestCase):
    def model(self):
        return plan(
            json.loads(
                (
                    ROOT
                    / "tests/fixtures/history/run-20260907T090013451671Z/schedule.json"
                ).read_text()
            )
        )

    def test_numerical_storage_and_explicit_rms_lease(self):
        r = self.model()
        self.assertEqual(r["validation"]["numerical_storage_bytes"], 7072)
        self.assertEqual(r["validation"]["phases"], 9)
        self.assertEqual(
            next(v for v in r["values"] if v["name"] == "public_x")["last"], 8
        )
        self.assertIn(dict(bank="src1", index=2), r["phases"][0]["acquire"]["square"])
        self.assertEqual(r["phases"][1]["acquire"]["row"], [dict(bank="src1", index=2)])

    def test_no_early_reuse_or_early_public_release(self):
        r = self.model()
        next(v for v in r["values"] if v["name"] == "down_then_final_residual")[
            "first"
        ] = 5
        with self.assertRaisesRegex(Error, "simultaneously live"):
            verify(r["storage"], r["values"], r["phases"])
        r = self.model()
        next(v for v in r["values"] if v["name"] == "public_x")["last"] = 7
        with self.assertRaisesRegex(Error, "immutable public"):
            verify(r["storage"], r["values"], r["phases"])

    def test_row_collective_waits_for_square_resources(self):
        r = self.model()
        r["phases"][0].update(release=[], join_before_next=False)
        with self.assertRaisesRegex(Error, "before completion"):
            verify(r["storage"], r["values"], r["phases"])
