"""Native early-s convergence and denominator-breakdown semantics."""

import subprocess, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


from host_compiler import executable as host_compiler

class BiCGStabReference(unittest.TestCase):
    def test_nonsymmetric_early_exit_and_zero_operator(self):
        code = r"""
#include "spatial.hpp"
#include <cassert>
int main(){
 spatial::tensor<7,1> a; a={{4,.5f,5,-.25f,6,1,7}};
 spatial::index_tensor<7,1> rows;rows={{0,0,1,1,2,2,3}};
 spatial::index_tensor<5,1> cols;cols={{0,1,3,5,7}};
 spatial::tensor<4,1> b,x;b={{4.5f,4.75f,7,7}};
 spatial::index_tensor<1,1> limit;limit.data[0]=16;
 spatial::tensor<2,1> tol;tol.data[0]=1e-5f;
 auto solved=spatial::bicgstab_csc<4,16>(a,rows,cols,b,x,limit,tol);
 assert(solved.reason.data[0]==0);for(float v:solved.solution.data)assert(std::abs(v-1)<1e-5f);
 a={{2,0,2,0,2,0,2}};b={{1,1,1,1}};
 auto early=spatial::bicgstab_csc<4,16>(a,rows,cols,b,x,limit,tol);
 assert(early.reason.data[0]==0 && early.iterations.data[0]==1 && early.true_residual_norm.data[0]==0);
 for(float v:early.solution.data)assert(v==.5f);
 a={{0,0,0,0,0,0,0}};
 auto zero=spatial::bicgstab_csc<4,16>(a,rows,cols,b,x,limit,tol);
 assert(zero.reason.data[0]==3 && zero.iterations.data[0]==0 && zero.true_residual_norm.data[0]==2);
 spatial::tensor<4,1> omat={{1,1,1,0}};
 spatial::index_tensor<4,1> orows={{0,1,0,1}};spatial::index_tensor<3,1> ocols={{0,2,4}};
 spatial::tensor<2,1> ob={{1,0}},ox;limit.data[0]=8;
 auto omega=spatial::bicgstab_csc<2,8>(omat,orows,ocols,ob,ox,limit,tol);
 assert(omega.reason.data[0]==3 && omega.iterations.data[0]==0 && omega.solution.data[0]==0 && omega.solution.data[1]==0 && omega.true_residual_norm.data[0]==1);

}
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "test.cpp").write_text(code)
            subprocess.run(
                [
                    host_compiler(),
                    "-std=c++17",
                    "-ffp-contract=off",
                    "-fsanitize=undefined",
                    "-fno-sanitize-recover=all",
                    "-I",
                    str(ROOT / "toolchain/include"),
                    str(p / "test.cpp"),
                    "-o",
                    str(p / "test"),
                ],
                capture_output=True,
                check=True,
            )
            subprocess.run([str(p / "test")], capture_output=True, check=True)


if __name__ == "__main__":
    unittest.main()
