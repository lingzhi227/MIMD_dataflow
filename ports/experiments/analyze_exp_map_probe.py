"""SDK exp microbenchmark: exact target bits and paired same-PE local intervals."""

import argparse, json, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, verify, sha

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from sdk_math_reference import exp_f16_nonpositive
from binary16 import bits

p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
verify(a.bundle)
e = read(a.bundle / "execution.json")
r = read(a.bundle / "results.json")
assert (
    e["success"]
    and e["results_sha256"] == sha(a.bundle / "results.json")
    and r["success"]
    and len(r["cases"]) == 3
)
reports = []
for epoch, c in enumerate(r["cases"]):
    x = np.asarray(c["input"], np.uint16).view(np.float16)
    expected = np.asarray(
        [bits(exp_f16_nonpositive(v)) for v in x.ravel()], np.uint16
    ).reshape(8, 512)
    np.testing.assert_array_equal(c["scalar"], expected)
    np.testing.assert_array_equal(c["mapped"], expected)
    np.testing.assert_array_equal(c["progress"], epoch + 1)
    t = np.asarray(c["timing"], np.int64)
    assert t.shape == (8, 9) and np.all((t >= 0) & (t < 65536))
    scalar = sum((t[:, i + 3] - t[:, i]) * (1 << (16 * i)) for i in range(3))
    mapped = sum((t[:, i + 6] - t[:, i + 3]) * (1 << (16 * i)) for i in range(3))
    assert np.all((scalar > 0) & (scalar < 2**32) & (mapped > 0) & (mapped < 2**32))
    reports.append(
        dict(
            target_bits_exact=True,
            scalar_cycles=scalar.tolist(),
            map_cycles=mapped.tolist(),
            max_scalar_cycles=int(scalar.max()),
            max_map_cycles=int(mapped.max()),
            max_interval_ratio_scalar_over_map=float(scalar.max() / mapped.max()),
        )
    )
out = dict(
    passed=True,
    new_sdk_execution=False,
    observed_output_words=24576,
    cases=reports,
    scope="EightPEs512values each, threechanged calls. Same SDKexp helper, explicitloop measuredfirst and@mapsecond within eachcall. Local simulator microbenchmark; no application or hardware speedup implied.",
    hashes={
        str(p): sha(p)
        for p in [
            a.bundle / "results.json",
            a.bundle / "provenance.json",
            ROOT / "toolchain/sdk_math_reference.py",
            Path(__file__),
        ]
    },
)
a.output.write_text(json.dumps(out, indent=2) + "\n")
print(a.output)
print(reports)
