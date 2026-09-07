"""Check explicit leases against actual CSL declarations and emitted bytes."""

import json, re, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from inference_resources import projection, compute, score_exchange, check_disjoint
from frontend import Error


class Resources(unittest.TestCase):
    def test_source_banks_and_fabric_reservations(self):
        src = (ROOT / "toolchain/runtime/inference_comm.csl").read_text()
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

        root = ROOT / "tests/fixtures/mlp-dsr-regression"
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
