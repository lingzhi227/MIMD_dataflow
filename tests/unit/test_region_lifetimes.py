"""Mutation tests for region buffer reuse and explicit completion contracts."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, sys, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from frontend import Error
from region_lifetimes import verify
from projection_residual_lifetimes import plan


class RegionLifetimes(unittest.TestCase):
    def test_actual_lowering_storage_accounting(self):
        for l, w, rows, features in ((64, 64, 8, 8), (512, 1024, 16, 32)):
            r = plan(l, w, rows, features)
            self.assertEqual(
                r["validation"]["numerical_storage_bytes"],
                2 * (5 * l + 3 * w + features + rows),
            )
            self.assertTrue(r["validation"]["checked"])

    def test_reject_overlapping_values_and_small_storage(self):
        for mutate in (
            lambda r: next(
                v for v in r["values"] if v["name"] == "projection_left_work"
            ).update(last=1),
            lambda r: r["storage"].update(XQ_tile=127),
            lambda r: r["values"][0].update(last=4),
        ):
            r = plan(64, 64, 8, 8)
            mutate(r)
            with self.assertRaises(Error):
                verify(r["storage"], r["values"], r["phases"])

    def test_reject_unjoined_projection_and_unknown_completion(self):
        for mutate in (
            lambda r: r["phases"][0]["release"].remove("left_fabric"),
            lambda r: r["phases"][1]["release"].append("not_started"),
            lambda r: r["phases"][0]["acquire"].update(
                conflict=[dict(bank="src1", index=3)]
            ),
        ):
            r = plan(64, 64, 8, 8)
            mutate(r)
            with self.assertRaises(Error):
                verify(r["storage"], r["values"], r["phases"])

    def test_local_library_must_release_before_row_collective(self):
        r = plan(64, 64, 8, 8)
        r["phases"][2].update(release=[], join_before_next=False)
        with self.assertRaisesRegex(Error, "before completion"):
            verify(r["storage"], r["values"], r["phases"])


if __name__ == "__main__":
    unittest.main()
