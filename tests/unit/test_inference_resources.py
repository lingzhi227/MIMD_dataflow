"""Check explicit leases against actual CSL declarations and emitted bytes."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import json, re, sys, tempfile, unittest
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from inference_resources import projection, compute, score_exchange, check_disjoint
from frontend import Error


class Resources(unittest.TestCase):
    def test_source_banks_and_fabric_reservations(self):
        src = (ROOT / "runtime/csl/inference_comm.csl").read_text()
        decl = {
            name: dict(bank=bank, index=int(index))
            for name, bank, index in re.findall(
                r"const\s+(\w+)\s*=\s*@get_dsr\(dsr_(\w+),\s*(\d+)\)", src
            )
        }
        names = dict(
            left_memory=["left_send_dsr", "left_recv_dsr"],
            right_memory=["right_send_dsr", "right_recv_dsr"],
            left_fabric=["left_matrix_out_dsr", "left_matrix_in_dsr"],
            right_fabric=["right_matrix_out_dsr", "right_matrix_in_dsr"],
        )
        self.assertEqual(
            projection(), {k: [decl[n] for n in ns] for k, ns in names.items()}
        )
        self.assertEqual(
            {v["index"] for group in score_exchange().values() for v in group}, {4, 6}
        )
        # Both endpoints are used by the async transfer, not merely declared.
        self.assertIn("@mov16(left_matrix_out_dsr, left_send_dsr,", src)
        self.assertIn("@mov16(right_recv_dsr, right_matrix_in_dsr,", src)

    def test_collision_is_bank_specific(self):
        check_disjoint(compute(), *[v for v in projection().values()])
        check_disjoint([dict(bank="src0", index=3)], [dict(bank="src1", index=3)])
        with self.assertRaises(Error):
            check_disjoint(projection()["left_memory"], [dict(bank="dest", index=3)])

    def test_metadata_correction_keeps_executed_csl(self):
        from mesh_mlp import plan, generate

        root = ROOT / "tests/fixtures/history/run-20260907T044447648813Z"
        m = json.loads((root / "semantic.json").read_text())
        old = json.loads((root / "schedule.json").read_text())
        new = plan(m)
        self.assertEqual(old["resources"]["communication_dsrs"], [3, 4])
        self.assertEqual(new["resources"]["communication_dsrs"], [3, 4, 5, 6])
        old.pop("resources")
        rest = dict(new)
        rest.pop("resources")
        self.assertEqual(old, rest)
        with tempfile.TemporaryDirectory() as td:
            generate(new, td)
            for p in Path(td).iterdir():
                self.assertEqual(p.read_bytes(), (root / p.name).read_bytes(), p.name)


if __name__ == "__main__":
    unittest.main()
