"""Fixed-budget completion, sign flips and safe zero normalization."""

import subprocess, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PowerReference(unittest.TestCase):
    def test_budget_sign_zero_and_tiny(self):
        source = r"""
#include "spatial.hpp"
#include <cassert>
int main(){
 spatial::tensor<4,1> a={{-2,-2,-2,-2}},x={{2,0,0,0}};
 spatial::index_tensor<4,1> rows={{0,1,2,3}};
 spatial::index_tensor<5,1> cols={{0,1,2,3,4}};
 spatial::index_tensor<1,1> steps;
 auto none=spatial::power_csc<4,8>(a,rows,cols,x,steps);
 assert(none.reason.data[0]==0 && none.iterations.data[0]==0 && none.vector.data[0]==2);
 steps.data[0]=3;auto odd=spatial::power_csc<4,8>(a,rows,cols,x,steps);
 assert(odd.reason.data[0]==0 && odd.iterations.data[0]==3 && odd.vector.data[0]==-1 && odd.norms.data[0]==4);
 steps.data[0]=4;auto even=spatial::power_csc<4,8>(a,rows,cols,x,steps);assert(even.vector.data[0]==1);
 a={{0,0,0,0}};auto zero=spatial::power_csc<4,8>(a,rows,cols,x,steps);
 assert(zero.reason.data[0]==1 && zero.iterations.data[0]==0 && zero.vector.data[0]==2);
 a={{1,1,1,1}};x={{0,0,0,0}};auto zv=spatial::power_csc<4,8>(a,rows,cols,x,steps);assert(zv.reason.data[0]==1 && zv.iterations.data[0]==0);
 x={{1e-30f,1e-30f,1e-30f,1e-30f}};steps.data[0]=1;auto tiny=spatial::power_csc<4,8>(a,rows,cols,x,steps);
 assert(tiny.reason.data[0]==0 && tiny.norms.data[0]>0 && std::abs(tiny.vector.data[0]-.5f)<1e-6f);
}
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "test.cpp").write_text(source)
            subprocess.run(
                [
                    "clang++",
                    "-std=c++17",
                    "-ffp-contract=off",
                    "-fsanitize=undefined",
                    "-I",
                    str(ROOT / "toolchain/include"),
                    str(p / "test.cpp"),
                    "-o",
                    str(p / "test"),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run([str(p / "test")], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
