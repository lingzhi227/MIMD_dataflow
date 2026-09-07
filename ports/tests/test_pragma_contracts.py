"""Declarative pragma compatibility, independent attribute order and diagnostics."""

import sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from pragma_contracts import parse, DATAFLOW_SCHEMAS
from frontend import parse as frontend


class PragmaContracts(unittest.TestCase):
    def test_every_schema_accepts_reordering_and_spacing(self):
        for schema in DATAFLOW_SCHEMAS:
            attrs = {
                k: ("4" if v == "uint" else v.split("|")[0]) for k, v in schema.items()
            }
            canonical = "#pragma csl dataflow " + " ".join(
                k + "=" + v for k, v in attrs.items()
            )
            reordered = "#pragma csl dataflow " + "  ".join(
                k + " = " + v for k, v in reversed(list(attrs.items()))
            )
            self.assertEqual(parse(canonical), parse(reordered))
        self.assertEqual(parse("#pragma csl place y = 3 x=2"), "place x=2 y=3")

    def test_reject_unknown_duplicate_missing_or_malformed(self):
        base = "#pragma csl dataflow rows=4 cols=4 broadcast=columns reduce=rows fp=relaxed compute=vector"
        for text in [
            base + " rows=4",
            base + " unknown=1",
            base.replace(" fp=relaxed", ""),
            base.replace("rows=4", "rows=-4"),
            base.replace("compute=vector", "compute=foo"),
            base.replace("cols=4", "cols=4.0"),
            "#pragma csl resident x=1",
            "#pragma csl place x=1 x=2",
            "#define rows 4",
        ]:
            with self.assertRaises(ValueError, msg=text):
                parse(text)
        with self.assertRaisesRegex(ValueError, "duplicate attribute rows"):
            parse(base + " rows=8")

    def test_frontend_ir_same_for_reordered_grouped_policy(self):
        source = (
            ROOT / "projects/waferllm/mesh_grouped_gemv_128_4x4_g2_f16/hls.cpp"
        ).read_text()
        line = next(v for v in source.splitlines() if v.startswith("#pragma"))
        changed = "#pragma csl dataflow " + " ".join(reversed(line.split()[3:]))
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hls.cpp"
            p.write_text(source)
            a = frontend(p)
            p.write_text(source.replace(line, changed))
            b = frontend(p)
            self.assertEqual(a["nodes"], b["nodes"])
            self.assertEqual(a["states"], b["states"])


if __name__ == "__main__":
    unittest.main()
