"""Actual native mixed-precision experiments, not registered HLS/CSL policies.

Preserve original inputs and numerical gates. Explore the connected V/PV/O/Z
precision region rather than changing data or declaring a looser contract.
"""

import datetime, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from native_transport import parse_outputs
from input_attention_fixtures import check, child_inputs
from attention_tail_fixtures import reference

HELPERS = r"""
namespace precision_experiment {
template<class O,int R,int K,int C,class A,class B>
auto product(const spatial::tensor<R,K,A>& a,const spatial::tensor<K,C,B>& b){
 spatial::tensor<R,C,O> out;
 for(int i=0;i<R;++i)for(int j=0;j<C;++j){
  float total=0;
  for(int k=0;k<K;++k)total=std::fma(float(a.data[i*K+k]),float(b.data[k*C+j]),total);
  out.data[i*C+j]=O(total);
 }return out;
}
template<class O,int R,int C,class A,class B>
auto add(const spatial::tensor<R,C,A>&a,const spatial::tensor<R,C,B>&b){
 spatial::tensor<R,C,O> out;
 for(int i=0;i<R*C;++i)out.data[i]=O(float(a.data[i])+float(b.data[i]));return out;
}
template<int R,int C,class A>
auto norm(const spatial::tensor<R,C,A>&x,const spatial::tensor<1,C,spatial::f16>&gamma,double epsilon){
 spatial::tensor<R,C,spatial::f16> out;
 for(int i=0;i<R;++i){double sum=0;for(int j=0;j<C;++j)sum+=double(x.data[i*C+j])*double(x.data[i*C+j]);
  double inv=1/std::sqrt(sum/C+epsilon);
  for(int j=0;j<C;++j)out.data[i*C+j]=spatial::f16(double(x.data[i*C+j])*double(gamma.data[j])*inv);
 }return out;
}
template<class O,int R,int C,class A>auto cast(const spatial::tensor<R,C,A>&x){
 spatial::tensor<R,C,O> out;for(int i=0;i<R*C;++i)out.data[i]=O(x.data[i]);return out;
}
}
"""


def study(baseline):
    baseline = Path(baseline).resolve()
    root = (
        ROOT
        / "evidence"
        / (
            "input-attention-mixed-precision-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    print(root.relative_to(ROOT), flush=True)
    bs = json.loads((baseline / "logical-inputs.json").read_text())
    report = []
    original = (baseline / "observed.cpp").read_text()
    for name, wide in (
        ("baseline", ()),
        ("wide-O-Z", ("projection",)),
        ("wide-PV-O-Z", ("attention", "projection")),
        ("wide-V-PV-O-Z", ("v", "attention", "projection")),
    ):
        dst = root / name
        dst.mkdir()
        text = original
        text = text.replace(
            '#include "spatial.hpp"', '#include "spatial.hpp"\n' + HELPERS
        )
        for variable, a, b in (
            ("v", "input_normalized", "v_weight"),
            ("attention", "probability", "v"),
            ("projection", "attention", "output_weight"),
        ):
            if variable in wide:
                old = f"auto {variable}=spatial::matmul({a},{b});"
                assert text.count(old) == 1
                text = text.replace(
                    old,
                    f"auto {variable}=precision_experiment::product<float>({a},{b});",
                )
        if wide:
            text = text.replace(
                "spatial::add(projection,input_x)",
                "precision_experiment::add<float>(projection,input_x)",
            )
            text = text.replace(
                "spatial::rmsnorm(z,gamma,0.000001)",
                "precision_experiment::norm(z,gamma,0.000001)",
            )
            text = text.replace(
                "spatial::add(z,delta)",
                "precision_experiment::add<spatial::f16>(z,delta)",
            )
        for variable in ("z", "x"):
            marker = " auto " + variable + "="
            start = text.index(marker)
            end = text.index(";", start) + 1
            text = (
                text[:end]
                + f'\n spatial::output("__detail_{variable}",{variable});\n'
                + text[end:]
            )
        marker = ' spatial::output("output",result);'
        text = text.replace(
            marker,
            ' spatial::output("__detail_projection_half",precision_experiment::cast<spatial::f16>(projection));\n'
            + marker,
        )
        (dst / "source.cpp").write_text(text)
        cmd = json.loads((baseline / "observed-command.json").read_text())
        cmd[cmd.index(str(baseline / "observed.cpp"))] = str(dst / "source.cpp")
        cmd[-1] = str(dst / "native")
        (dst / "command.json").write_text(json.dumps(cmd) + "\n")
        result = subprocess.run(cmd, capture_output=True, text=True)
        (dst / "compile.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0
        result = subprocess.run(
            [cmd[-1]],
            input=(baseline / "native-input.txt").read_text(),
            capture_output=True,
            text=True,
        )
        (dst / "output.txt").write_text(result.stdout)
        (dst / "stderr.txt").write_text(result.stderr)
        assert result.returncode == 0
        cases = []
        for epoch, (b, row) in enumerate(zip(bs, parse_outputs(result.stdout))):
            obs = {k[10:]: v for k, v in row.items() if k.startswith("__observe_")}
            obs["v"] = obs["v_raw"]
            _, child = child_inputs(64, 64, 1e-6, b)
            expected = reference(64, 64, 256, 1e-6, 0.125, child)
            branches = {}
            for key, wanted in zip(
                (
                    "score",
                    "probability",
                    "attention",
                    "projection",
                    "z",
                    "delta",
                    "output",
                ),
                expected,
            ):
                actual = np.array(
                    row["output"]
                    if key == "output"
                    else row["__detail_z"] if key == "z" else obs[key]
                ).reshape(wanted.shape)
                err = actual - wanted
                branches[key] = dict(
                    relative_l2=float(
                        np.linalg.norm(err) / max(np.linalg.norm(wanted), 1e-30)
                    ),
                    peak_scaled_error=float(
                        np.max(np.abs(err)) / max(np.max(np.abs(wanted)), 1e-30)
                    ),
                    expected_range=[float(wanted.min()), float(wanted.max())],
                    actual_range=[float(actual.min()), float(actual.max())],
                )
            try:
                numeric = check(
                    64, 64, 256, 1e-6, 0.125, b, {"output": row["output"]}, obs
                )
                passed = True
                error = None
            except AssertionError as e:
                numeric = None
                passed = False
                error = repr(e)
            cases.append(
                dict(
                    epoch=epoch,
                    passed=passed,
                    gate_error=error,
                    numerical=numeric,
                    branches=branches,
                )
            )
        report.append(
            dict(
                policy=name,
                wide_values=list(wide),
                passed=all(v["passed"] for v in cases),
                cases=cases,
            )
        )
        print(name, [c["passed"] for c in cases], flush=True)
    (root / "review.json").write_text(
        json.dumps(dict(scope=__doc__, policies=report), indent=2) + "\n"
    )
    (root / "driver.py").write_bytes(Path(__file__).read_bytes())
    (root / "baseline.json").write_text(
        json.dumps(
            dict(
                path=str(baseline.relative_to(ROOT)),
                files={
                    str(p.relative_to(baseline)): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in baseline.rglob("*")
                    if p.is_file()
                    and (
                        p.suffix in (".hpp", ".cpp")
                        or p.name in ("native-input.txt", "logical-inputs.json")
                    )
                },
            ),
            indent=2,
        )
        + "\n"
    )
    return root


if __name__ == "__main__":
    study(sys.argv[1])
