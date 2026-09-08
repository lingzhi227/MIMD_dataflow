"""Check a sufficient gating error inequality using exhaustive observed SDK SiLU."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, math
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify

p = argparse.ArgumentParser()
p.add_argument("probe", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
verify(a.probe)
e = read(a.probe / "execution.json")
assert e["success"] and e["results_sha256"] == sha(a.probe / "results.json")
c = read(a.probe / "results.json")["cases"][0]
raw = np.asarray(c["input"], np.uint16).ravel()
assert sorted(raw.tolist()) == list(range(0x7C00))
magnitude = raw.view(np.float16).astype(float)
select = magnitude <= 8
rows = []
for sign, key in [(-1, "negative_silu"), (1, "positive_silu")]:
    g = sign * magnitude[select]
    observed = (
        np.asarray(c[key], np.uint16).ravel().view(np.float16).astype(float)[select]
    )
    std = np.array([float(v) / (1 + math.exp(-float(v))) for v in g])
    assert np.all(np.isfinite(observed))
    # RN half product: abs(fl(u*a)-u*a) <=2^-11*abs(u*a)+2^-25.
    # Subtract advertised .004*abs(u*s)+2^-24*(1+abs(u));
    # worst |u| in[0,8] occurs at0 or8 since this bound is affine.
    coefficient = (
        np.abs(observed - std)
        + 2**-11 * np.abs(observed)
        - 0.004 * np.abs(std)
        - 2**-24
    )
    excess = np.maximum(0, 8 * coefficient) - 2**-25
    assert np.all(excess < 0)
    rows.append(
        dict(
            sign=sign,
            gate_encodings=len(g),
            max_sufficient_error_excess=float(excess.max()),
            max_activation=float(np.max(np.abs(observed))),
            worst_gate=float(g[np.argmax(excess)]),
        )
    )
a.output.write_text(
    json.dumps(
        dict(
            passed=True,
            new_sdk_execution=False,
            scope="Every finite half gate of magnitude <=8, both signed zero encodings included, using actual exhaustive SDK SiLU observations and double math.exp reference. Sufficient inequality for all real |up|<=8 conditional on RN binary16 multiply bound; not exhaustive enumeration of every up/gate device pair or a formal real-analysis proof. No overflow: |up*activation|<=64.",
            derivation="For a=SDK SiLU(g), s=standard SiLU(g), product error <=|up|*(|a-s|+2^-11*|a|)+2^-25. Compare against .004*|up*s|+2^-24*(1+|up|); affine maximum over |up| in[0,8].",
            cases=rows,
            hashes={
                str(v): sha(v)
                for v in [
                    a.probe / "provenance.json",
                    a.probe / "results.json",
                    Path(__file__),
                ]
            },
        ),
        indent=2,
    )
    + "\n"
)
print(a.output)
print(rows)
