"""Input domains must be executable and survive frontend extraction."""

import copy, json, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from frontend import parse, Error
from input_contracts import (
    verify_declarations,
    validate_batch,
    half_dot_bound,
    half_operand_bound,
)


from host_compiler import executable as host_compiler

class InputContracts(unittest.TestCase):
    def test_frontend_optional_bound_and_legacy_nodes(self):
        original = ROOT / "projects/waferllm/attention_64x128_8x8/hls.cpp"
        baseline = parse(original)
        frozen = json.loads(
            (
                ROOT / "tests/fixtures/history/run-20260907T032800549283Z/01_frontend_ir.json"
            ).read_text()
        )
        self.assertEqual(baseline["nodes"], frozen["nodes"])
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bounded.cpp"
            p.write_text(original.read_text().replace('("q");', '("q",0.125);'))
            bounded = parse(p)
            self.assertEqual(bounded["nodes"][0]["abs_bound"], 0.125)
            self.assertEqual(bounded["nodes"][1:], baseline["nodes"][1:])
        verify_declarations(bounded)
        bounded["input_bound"] = 1
        with self.assertRaises(Error):
            validate_batch(bounded, {"q": [0.25]})
        validate_batch(bounded, {"q": [0.125, -0.125, 0]})
        broken = copy.deepcopy(bounded)
        broken["nodes"][0]["abs_bound"] = float("nan")
        with self.assertRaises(Error):
            verify_declarations(broken)

    def test_native_input_contract_rejects_bad_value(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.cpp"
            p.write_text("""#include "spatial.hpp"
int main(){spatial::inputs["x"]={0.125f};auto a=spatial::input<1,1,spatial::f16>("x",0.125);if(a.data[0]!=0.125)return 1;spatial::inputs["x"]={0.25f};try{spatial::input<1,1,spatial::f16>("x",0.125);}catch(const std::runtime_error&){return 0;}return 2;}
""")
            exe = Path(td) / "test"
            subprocess.run(
                [
                    host_compiler(),
                    "-std=c++17",
                    "-I",
                    str(ROOT / "toolchain/include"),
                    str(p),
                    "-o",
                    str(exe),
                ],
                check=True,
                capture_output=True,
            )
            self.assertEqual(subprocess.run([str(exe)]).returncode, 0)

    def test_half_range_recurrence_and_overflow(self):
        self.assertEqual(half_dot_bound(0.125, 0.125, 64), 1)
        self.assertEqual(half_dot_bound(0.125, 0.125, 128), 2)
        self.assertLessEqual(half_operand_bound(0.1), 0.1)
        self.assertGreater(half_operand_bound(0.1), 0.0999)
        with self.assertRaises(Error):
            half_dot_bound(65504, 2, 1)


if __name__ == "__main__":
    unittest.main()
