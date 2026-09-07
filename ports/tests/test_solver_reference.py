"""Native solver result/termination contract; distributed lowering is separate."""

import subprocess, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SolverReference(unittest.TestCase):
    def test_cg_reasons_and_true_residual(self):
        source = r"""
#include "spatial.hpp"
#include <cassert>
int main(){
 spatial::tensor<4,1> a,b,x;
 spatial::index_tensor<4,1> rows;
 spatial::index_tensor<5,1> offsets;
 for(int i=0;i<4;++i){a.data[i]=float(i+1);b.data[i]=1;rows.data[i]=i;offsets.data[i]=i;}
 offsets.data[4]=4;
 spatial::index_tensor<1,1> limit;limit.data[0]=8;
 spatial::tensor<2,1> tol;tol.data[0]=1e-5f;
 auto solved=spatial::cg_csc<4,8>(a,rows,offsets,b,x,limit,tol);
 assert(solved.reason.data[0]==0 && solved.iterations.data[0]>0);
 for(int i=0;i<4;++i) assert(std::abs(solved.solution.data[i]-1.0f/(i+1))<1e-5f);
 assert(solved.true_residual_norm.data[0]<=2e-5f);
 limit.data[0]=1;
 auto capped=spatial::cg_csc<4,8>(a,rows,offsets,b,x,limit,tol);
 assert(capped.reason.data[0]==1 && capped.iterations.data[0]==1);
 limit.data[0]=0;
 auto none=spatial::cg_csc<4,8>(a,rows,offsets,b,x,limit,tol);
 assert(none.reason.data[0]==1 && none.iterations.data[0]==0);
 limit.data[0]=8;
 for(int i=0;i<4;++i){a.data[i]=1;x.data[i]=1;}
 auto initial=spatial::cg_csc<4,8>(a,rows,offsets,b,x,limit,tol);
 assert(initial.reason.data[0]==0 && initial.iterations.data[0]==0);
 for(int i=0;i<4;++i){b.data[i]=0;x.data[i]=0;}
 auto zero=spatial::cg_csc<4,8>(a,rows,offsets,b,x,limit,tol);
 assert(zero.reason.data[0]==0 && zero.true_residual_norm.data[0]==0);
 for(int i=0;i<4;++i){a.data[i]=-1;b.data[i]=1;}
 auto bad=spatial::cg_csc<4,8>(a,rows,offsets,b,x,limit,tol);
 assert(bad.reason.data[0]==2 && bad.iterations.data[0]==0);
 for(int i=0;i<4;++i){a.data[i]=1;b.data[i]=1e-30f;}
 auto tiny=spatial::cg_csc<4,8>(a,rows,offsets,b,x,limit,tol);
 assert(tiny.reason.data[0]==3 && tiny.iterations.data[0]==0 && tiny.true_residual_norm.data[0]>0);
 for(int i=0;i<4;++i){a.data[i]=(i%2==0)?1.0f:std::nextafter(1.0f,2.0f);b.data[i]=1e-20f;}
 tol.data[0]=1e-10f;
 auto later=spatial::cg_csc<4,8>(a,rows,offsets,b,x,limit,tol);
 assert(later.reason.data[0]==3 && later.iterations.data[0]==1 && later.true_residual_norm.data[0]>0);
 for(int i=0;i<4;++i){a.data[i]=float(1<<i);b.data[i]=1;x.data[i]=0;}
 tol.data[0]=1e-5f;
 auto pcg=spatial::pcg_csc<4,8>(a,rows,offsets,b,x,limit,tol);
 assert(pcg.reason.data[0]==0 && pcg.iterations.data[0]==1 && pcg.true_residual_norm.data[0]==0);
 for(int i=0;i<4;++i) assert(pcg.solution.data[i]==1.0f/a.data[i]);
 a.data[0]=0;bool invalid_diag=false;
 try{spatial::pcg_csc<4,8>(a,rows,offsets,b,x,limit,tol);}catch(const std::runtime_error&){invalid_diag=true;}
 assert(invalid_diag);
 limit.data[0]=9;bool rejected=false;
 try{spatial::cg_csc<4,8>(a,rows,offsets,b,x,limit,tol);}catch(const std::runtime_error&){rejected=true;}
 assert(rejected);
}
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test.cpp").write_text(source)
            subprocess.run(
                [
                    "clang++",
                    "-std=c++17",
                    "-ffp-contract=off",
                    "-fsanitize=undefined",
                    "-fno-sanitize-recover=all",
                    "-I",
                    str(ROOT / "toolchain/include"),
                    str(root / "test.cpp"),
                    "-o",
                    str(root / "test"),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run([str(root / "test")], check=True, capture_output=True)

    def test_integer_result_transport_preserves_u32(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test.cpp").write_text(
                '#include "spatial.hpp"\nvoid design(){auto x=spatial::index_input<2,1>("x");spatial::output("result",x);}'
            )
            subprocess.run(
                [
                    "clang++",
                    "-std=c++17",
                    "-DMW_EPOCHS=1",
                    "-DMW_BOUND=64",
                    "-I",
                    str(ROOT / "toolchain/include"),
                    str(root / "test.cpp"),
                    str(ROOT / "toolchain/runtime/native.cpp"),
                    "-o",
                    str(root / "test"),
                ],
                check=True,
                capture_output=True,
            )
            result = subprocess.run(
                [str(root / "test")],
                input="1\n1\n@u32 x 2 16777217 4294967295\n",
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(
                result.stdout, "epoch 0\n@u32 result 2 16777217 4294967295\n"
            )


if __name__ == "__main__":
    unittest.main()
