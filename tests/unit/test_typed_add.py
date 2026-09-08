"""Residual addition retains tensor dtype and rounds once in binary16."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from frontend import parse


from host_compiler import executable as host_compiler

class TypedAdd(unittest.TestCase):
    def test_native_half_ties_cancellation_and_subnormals(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "native.cpp"
            exe = Path(td) / "native"
            p.write_text("""#include "spatial.hpp"
int main(){spatial::tensor<1,4,spatial::f16>a,b;
float x[]={1,1.0009765625,0.000000059604644775390625,65504};
float y[]={0.00048828125,0.00048828125,0.000000059604644775390625,-65504};
for(int i=0;i<4;++i){a.data[i]=x[i];b.data[i]=y[i];}
auto c=spatial::add(a,b);static_assert(std::is_same_v<decltype(c),spatial::tensor<1,4,spatial::f16>>);
for(auto v:c.data)std::cout<<std::setprecision(9)<<float(v)<<" ";}
""")
            subprocess.run(
                [
                    host_compiler(),
                    "-std=c++17",
                    "-I",
                    str(ROOT / "include/pragma"),
                    str(p),
                    "-o",
                    str(exe),
                ],
                check=True,
                capture_output=True,
            )
            actual = np.asarray(
                subprocess.check_output([str(exe)], text=True).split(), np.float32
            )
            expected = np.asarray([1, 1.001953125, 2**-23, 0], np.float32)
            np.testing.assert_array_equal(actual, expected)

    def test_frontend_preserves_half_add(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "hls.cpp"
            p.write_text("""#include "spatial.hpp"
void design(){auto x=spatial::input<8,8,spatial::f16>("x",0.125);auto y=spatial::input<8,8,spatial::f16>("y",0.125);auto z=spatial::add(x,y);spatial::output("z",z);}
""")
            m = parse(p)
            n = next(v for v in m["nodes"] if v["op"] == "add")
            self.assertEqual(n["dtype"], "f16")
            self.assertEqual(n["shape"], [8, 8])
            self.assertEqual(len(n["inputs"]), 2)


if __name__ == "__main__":
    unittest.main()
