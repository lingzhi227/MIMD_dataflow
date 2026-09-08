"""Half FMA semantics and fail-closed frontend precision boundaries."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import sys, subprocess, tempfile, unittest
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from binary16 import bits, fma
from frontend import parse
from ir import verify


from host_compiler import executable as host_compiler

class Binary16(unittest.TestCase):
    def test_bit_mismatch_points_to_lane_and_values(self):
        from binary16 import assert_bits_equal

        assert_bits_equal([0, 15360], [0, 15360], "same")
        with self.assertRaisesRegex(
            ValueError, r"PE3 round2: first mismatch at \(1,\).*0x3c01.*0x3c00"
        ):
            assert_bits_equal([0, 15361], [0, 15360], "PE3 round2")
        with self.assertRaisesRegex(ValueError, "first mismatch"):
            assert_bits_equal(15361, 15360, "scalar")
        with self.assertRaisesRegex(ValueError, "shapes"):
            assert_bits_equal([0], [[0]], "shape")

    def test_native_fused_rounding(self):
        source = r"""
#include "spatial.hpp"
#include <cassert>
int main(){
 spatial::tensor<1,2,spatial::f16>a={{-1.0009765625,1.0009765625}};
 spatial::tensor<2,1,spatial::f16>b={{1,1.0009765625}};
 auto c=spatial::matmul(a,b);
 assert(float(c.data[0])==0.00097751617431640625f);
 spatial::inputs["x"]={1.0009765625f};auto x=spatial::input<1,1,spatial::f16>("x");
 spatial::output("x",x);assert(spatial::outputs.at("x")[0]==1.0009765625f);
}
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "test.cpp").write_text(source)
            subprocess.run(
                [
                    host_compiler(),
                    "-std=c++17",
                    "-ffp-contract=off",
                    "-fsanitize=undefined",
                    "-I",
                    str(ROOT / "include/pragma"),
                    str(p / "test.cpp"),
                    "-o",
                    str(p / "test"),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run([str(p / "test")], check=True, capture_output=True)
        self.assertEqual(
            fma(1.0009765625, 1.0009765625, -1.0009765625), 0.00097751617431640625
        )
        self.assertEqual(bits(fma(2**-24, 1, 0)), 1)
        self.assertEqual(bits(fma(2**-14, 2**-11, 0)), 0)
        self.assertEqual(fma(65504, 2, -65504), 65504)

    def test_unqualified_half_graph_rejected(self):
        source = """#include "spatial.hpp"
void design(){
 auto a=spatial::input<2,2,spatial::f16>("a");
 auto b=spatial::input<2,2,spatial::f16>("b");
 auto c=spatial::matmul(a,b);spatial::output("result",c);
}
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "source.cpp"
            p.write_text(source)
            m = parse(p)
        self.assertEqual([n.get("dtype") for n in m["nodes"][:3]], ["f16"] * 3)
        with self.assertRaisesRegex(ValueError, "f16 requires"):
            verify(m, 2, 2)


if __name__ == "__main__":
    unittest.main()
