"""Exercise integer source/IR/native ABI, including values unrepresentable in f32."""

import subprocess
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from frontend import parse
from ir import verify


from host_compiler import executable as host_compiler

class SparseFrontend(unittest.TestCase):
    def test_sparse_ir_types_and_capacity(self):
        raw = parse(ROOT / "projects/sdk_examples/mesh_spmv_512x512_4x4/hls.cpp")
        module = verify(raw, 4, 64)
        self.assertEqual(
            [n["dtype"] for n in module["nodes"]],
            ["f32", "u32", "u32", "f32", "f32", "f32"],
        )
        self.assertEqual(module["profile"], "mesh_spmv.v1")
        raw["nodes"][4]["dataflow"]["nnz_per_pe"] = 65535
        with self.assertRaises(ValueError):
            verify(raw, 4, 64)

    def test_joint_resource_ownership(self):
        import copy
        from mesh_spmv import resource_contract, validate_resources

        original = resource_contract()
        for category in ("queue", "microthread", "dsr", "task", "task_low", "color"):
            r = copy.deepcopy(original)
            if category == "queue":
                r["spmv_input_queues"][1] = 1
            elif category == "microthread":
                r["phase_microthreads"]["north_south"]["send"][0] = 2
            elif category == "dsr":
                r["dest_dsr"][1] = r["dest_dsr"][0]
            elif category == "task_low":
                r["local_tasks"][0] = 7
            elif category == "color":
                r["colors"][0] = 23
            else:
                r["local_tasks"][0] = 29
            with self.subTest(category=category), self.assertRaises(ValueError):
                validate_resources(r)

    def test_native_integer_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "probe.cpp").write_text("""#include "spatial.hpp"
void design(){
 auto i=spatial::index_input<1,1>("indices");
 spatial::require(i.data[0]==16777217u || i.data[0]==4294967295u);
 spatial::tensor<1,1> result;result.data[0]=1;
 spatial::output("result",result);
}
""")
            subprocess.run(
                [
                    host_compiler(),
                    "-std=c++17",
                    "-DMW_EPOCHS=1",
                    "-DMW_BOUND=64",
                    "-fsanitize=undefined",
                    "-fno-sanitize-recover=all",
                    "-I",
                    str(ROOT / "toolchain/include"),
                    str(root / "probe.cpp"),
                    str(ROOT / "toolchain/runtime/native.cpp"),
                    "-o",
                    str(root / "probe"),
                ],
                check=True,
                capture_output=True,
            )
            for token, accepted in [
                ("16777217", True),
                ("4294967295", True),
                ("16777216", False),
                ("16777217.0", False),
                ("-1", False),
                ("4294967296", False),
                ("1e2", False),
            ]:
                with self.subTest(token=token):
                    result = subprocess.run(
                        [str(root / "probe")],
                        input="1\n1\n@u32 indices 1 " + token + "\n",
                        text=True,
                        capture_output=True,
                    )
                    self.assertEqual(result.returncode == 0, accepted)
                    if accepted:
                        self.assertIn("result 1 1", result.stdout)


if __name__ == "__main__":
    unittest.main()
