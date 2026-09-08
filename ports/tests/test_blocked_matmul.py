"""Explicit arithmetic, native tails, physical block compatibility and precision."""

import sys, tempfile, subprocess, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain"), str(ROOT / "experiments")]
from blocked_matmul import evaluate
from binary16 import matmul
from frontend import parse, Error
from ir import verify
from mesh_mlp import plan
from build_mlp_profiles import source


from host_compiler import executable as host_compiler

class BlockedMatmul(unittest.TestCase):
    def test_two_level_semantics_and_short_last_block(self):
        a = np.asarray([[2048, 1, -2048, 1, 2, -2, 2**-10]], float)
        b = np.ones((7, 1))
        self.assertEqual(float(matmul(a, b)[0, 0]), 1 + 2**-10)
        self.assertEqual(float(evaluate(a, b, 1)[0, 0]), float(np.float16(2 + 2**-10)))
        expected = evaluate(a, b, 3)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.cpp"
            p.write_text("""#include "spatial.hpp"
int main(){spatial::tensor<1,7,spatial::f16>a;spatial::tensor<7,1,spatial::f16>b;float v[]={2048,1,-2048,1,2,-2,0.0009765625};for(int i=0;i<7;++i){a.data[i]=v[i];b.data[i]=1;}auto c=spatial::matmul_blocked<spatial::f16,spatial::scalar>(a,b,3);std::cout<<std::setprecision(9)<<float(c.data[0]);}
""")
            exe = Path(td) / "native"
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
            self.assertEqual(
                float(
                    np.float32(float(subprocess.check_output([str(exe)], text=True)))
                ),
                float(expected[0, 0]),
            )

    def module(self, text=None):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hls.cpp"
            p.write_text(text or source(128, 128, 512, 8, True))
            m = parse(p)
            m["instrumentation"] = "counters"
            return m

    def test_frontend_and_physical_contract(self):
        m = verify(self.module(), 6, 1)
        down = m["nodes"][8]
        self.assertEqual(down["op"], "matmul")
        self.assertEqual(down["block_size"], 64)
        self.assertEqual(down["accumulation"], "block_f32")
        s = plan(m)
        self.assertEqual(s["down_block_size"], 64)
        self.assertEqual(
            s["memory_per_pe"]["wide_accumulator_and_conversion_scratch"], 12 * 256
        )

    def test_no_silent_policy_change_or_unsupported_tail(self):
        for text in [
            source(128, 128, 512, 8, True).replace("(hidden,d,64)", "(hidden,d,63)"),
            source(128, 128, 512, 8, True).replace(" accumulation=block_f32", ""),
            source(128, 128, 512, 8, True).replace(
                "spatial::matmul_blocked<spatial::f16,spatial::scalar>(hidden,d,64)",
                "spatial::matmul(hidden,d)",
            ),
        ]:
            with self.assertRaises(Error):
                verify(self.module(text), 6, 1)

    def test_uniform_failure_is_improved_without_changing_limit(self):
        # Gate/up are2. Isolate final contraction with target-rounded hidden.
        from sdk_math_reference import silu_f16

        h = float(np.float16(2 * silu_f16(2)))
        a = np.full((1, 512), h)
        b = np.full((512, 1), 0.125)
        nominal = 512 * 0.125 * 2 * (2 / (1 + np.exp(-2)))
        self.assertGreater(abs(float(matmul(a, b)[0, 0]) - nominal) / nominal, 0.02)
        self.assertLess(abs(float(evaluate(a, b, 64)[0, 0]) - nominal) / nominal, 0.02)


if __name__ == "__main__":
    unittest.main()
