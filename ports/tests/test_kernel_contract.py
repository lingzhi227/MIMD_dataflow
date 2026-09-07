"""Semantic regression tests for the shared frontend, not algorithm-name templates."""

import hashlib, json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from frontend import parse, Error
from local_kernel import evaluate_kernel, emit_kernel
from compile import build


class KernelContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def source(self, body):
        p = self.root / "test.cpp"
        p.write_text(
            '#include "spatial.hpp"\nvoid design(){auto a=spatial::input<1,1>("a");auto y=spatial::kernel(a,[](const spatial::tensor<1,1>&a){spatial::tensor<1,1> out{};'
            + body
            + 'return out;});spatial::output("y",y);}\n'
        )
        return p

    def kernel(self, body):
        return next(
            n["body"] for n in parse(self.source(body))["nodes"] if n["op"] == "kernel"
        )

    def test_pinned_reference_integrity(self):
        inventory = json.loads((ROOT / "evidence/source_inventory.json").read_text())
        for project, entry in inventory.items():
            for source in entry["files"]:
                p = ROOT / "projects" / project / "upstream" / source["path"]
                with self.subTest(path=str(p)):
                    self.assertEqual(
                        hashlib.sha256(p.read_bytes()).hexdigest(), source["sha256"]
                    )

    def test_reject_early_return(self):
        with self.assertRaisesRegex(Error, "early/nested"):
            self.kernel("if(a.data[0]>0.0f){return out;}")

    def test_reject_lossy_casts(self):
        for body in (
            "out.data[0]=a.data[0]+0.5;",
            "int x=a.data[0];out.data[0]=x;",
            "out.data[0]=1u;",
        ):
            with self.subTest(body=body), self.assertRaises(Error):
                self.kernel(body)

    def test_boolean_and_short_circuit(self):
        body = "bool flag=true;if(flag && !(a.data[0]<0.0f)){out.data[0]=a.data[0];}"
        k = self.kernel(body)
        self.assertEqual(evaluate_kernel(k, [[2.0]]), [2.0])
        csl, _ = emit_kernel(k)
        self.assertIn("=true;", csl)
        self.assertIn("(!", csl)
        build(
            self.source(body),
            self.root / "build",
            epochs=1,
            bound=4,
            batches=[{"a": [2.0]}],
        )

    def test_bounds_before_memory_access(self):
        k = self.kernel("int i=2;out.data[0]=a.data[i];")
        with self.assertRaisesRegex(Error, "array bounds"):
            evaluate_kernel(k, [[1.0]])

    def test_loop_guard(self):
        k = self.kernel("for(int i=0;i<257;++i){out.data[0]+=1.0f;}")
        with self.assertRaisesRegex(Error, "loop bound"):
            evaluate_kernel(k, [[1.0]])

    def test_failed_frontend_keeps_evidence(self):
        dest = self.root / "bad"
        with self.assertRaises(Error):
            build(self.source("out.data[0]=a.data[0]+0.5;"), dest)
        self.assertEqual(
            json.loads((dest / "stage.json").read_text())["stage"], "frontend"
        )
        self.assertTrue((dest / "00_clang_ast.json").exists())

    def test_odd_wire_padding_is_separate_from_tensor_extent(self):
        out = build(
            self.source("out.data[0]=a.data[0]*2.0f;"),
            self.root / "pad",
            epochs=1,
            bound=4,
            batches=[{"a": [2.0]}],
        )
        s = json.loads((out / "schedule.json").read_text())
        for n in s["nodes"]:
            self.assertEqual(n["output_size"], 1)
            self.assertEqual(n["wire_output_size"], 2)
        self.assertEqual(
            json.loads((out / "reference.json").read_text())["outputs"], [{"y": [4.0]}]
        )


if __name__ == "__main__":
    unittest.main()
