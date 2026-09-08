"""Explicit storage/compute types must survive Clang and never fall through."""

import copy, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "toolchain"), str(ROOT / "experiments")]
from frontend import parse, Error
from ir import verify
from host_compiler import executable
from input_attention_mixed_source import source


class ExplicitPrecisionFrontend(unittest.TestCase):
    def test_mixed_graph_records_precision_and_selects_scoped_backend(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hls.cpp"
            p.write_text(source())
            m = parse(p)
        expected = dict(
            v="f32",
            probability="f32",
            attention="f32",
            projection="f32",
            z="f32",
            x="f16",
            result="f16",
        )
        actual = {n["id"]: n for n in m["nodes"] if "precision" in n}
        self.assertEqual(set(actual), set(expected))
        for name, dtype in expected.items():
            self.assertEqual(
                actual[name]["precision"],
                dict(compute="f32", storage=dtype, explicit=True),
            )
            self.assertEqual(actual[name]["dtype"], dtype)
        for name in ("v", "attention", "projection"):
            self.assertEqual(actual[name]["accumulation"], "f32")
            self.assertEqual(actual[name]["dataflow"]["accumulation"], "f32")
        m["instrumentation"] = "counters"
        self.assertEqual(verify(m, 8, 2)["profile"], "mesh_input_attention_mixed.v1")

    def test_precision_agreement_is_checked_before_layout(self):
        from precision_contracts import verify as precision_verify

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hls.cpp"
            p.write_text(source())
            m = parse(p)
        facts = precision_verify(m)
        self.assertEqual(len(facts), 7)
        self.assertEqual([r["node"] for r in facts if r["narrowing"]], ["x", "result"])
        bad = copy.deepcopy(m)
        next(n for n in bad["nodes"] if n["id"] == "v")["dataflow"][
            "accumulation"
        ] = "block_f32"
        with self.assertRaisesRegex(Error, "pragma accumulation agreement"):
            precision_verify(bad)
        bad = copy.deepcopy(m)
        next(n for n in bad["nodes"] if n["id"] == "probability")["precision"][
            "storage"
        ] = "f16"
        with self.assertRaisesRegex(Error, "matching result storage"):
            precision_verify(bad)

    def test_native_explicit_f32_accumulation_changes_rounding(self):
        text = """#include "spatial.hpp"
int main(){
 spatial::tensor<1,3,spatial::f16> a; a.data[0]=2048;a.data[1]=1;a.data[2]=-2048;
 spatial::tensor<3,1,spatial::f16> b;b.data[0]=b.data[1]=b.data[2]=1;
 auto old=spatial::matmul(a,b);
 auto wide=spatial::matmul<spatial::scalar,spatial::scalar>(a,b);
 auto narrow=spatial::matmul<spatial::f16,spatial::scalar>(a,b);
 return old.data[0]==0 && wide.data[0]==1 && narrow.data[0]==1 ? 0 : 1;
}"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "main.cpp"
            p.write_text(text)
            out = Path(td) / "native"
            r = subprocess.run(
                [
                    executable(),
                    "-std=c++17",
                    "-ffp-contract=off",
                    "-I",
                    str(ROOT / "toolchain/include"),
                    str(p),
                    "-o",
                    str(out),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(subprocess.run([str(out)]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
