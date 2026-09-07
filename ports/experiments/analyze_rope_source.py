"""Compare actual original pair transform to source-order and standard rotations."""

import argparse, json
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify

p = argparse.ArgumentParser()
p.add_argument("bundles", type=Path, nargs="+")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
reports = []
for root in a.bundles:
    verify(root)
    e = read(root / "execution.json")
    assert e["success"] and e["results_sha256"] == sha(root / "results.json")
    s = read(root / "schema.json")
    m, n = s["Mt"], s["Nt"]
    r = read(root / "results.json")
    assert len(r["cases"]) == 3
    cases = []
    q = lambda v: np.asarray(v, np.float16).astype(float)
    for i, (b, c) in enumerate(zip(read(root / "inputs.json"), r["cases"])):
        x = np.asarray(b[: m * n]).reshape(m, n, order="F")
        sn = np.asarray(b[m * n : m * n + n // 2])
        co = np.asarray(b[m * n + n // 2 :])
        actual = (
            np.asarray(c["result"], np.uint16)
            .view(np.float16)
            .astype(float)[: m * n]
            .reshape(m, n, order="F")
        )
        even, odd = x[:, ::2], x[:, 1::2]
        source = np.empty_like(x)
        source[:, ::2] = q(q(odd * co) - q(even * sn))
        source[:, 1::2] = q(q(even * co) + q(odd * sn))
        standard = np.empty_like(x)
        standard[:, ::2] = q(q(even * co) - q(odd * sn))
        standard[:, 1::2] = q(q(odd * co) + q(even * sn))
        source_mismatch = int(
            np.count_nonzero(
                actual.astype(np.float16).view(np.uint16)
                != source.astype(np.float16).view(np.uint16)
            )
        )
        cases.append(
            dict(
                epoch=i,
                source_pair_formula_mismatches=source_mismatch,
                standard_pair_rotation_mismatches=int(
                    np.count_nonzero(actual != standard)
                ),
                zero_angle_identity_mismatches=(
                    int(np.count_nonzero(actual != x)) if i == 0 else None
                ),
                max_source_formula_error=float(np.max(np.abs(actual - source))),
            )
        )
        if m == n // 2 or s.get("row_scratch", False):
            assert source_mismatch == 0
    reports.append(
        dict(
            bundle=str(root),
            shape=[m, n],
            temporary_dsd_length=m if s.get("row_scratch", False) else n // 2,
            original_temporary_dsd_length=n // 2,
            row_dsd_length=m,
            cases=cases,
            results_sha256=sha(root / "results.json"),
        )
    )
a.output.write_text(
    json.dumps(
        dict(
            passed=True,
            new_sdk_execution=False,
            scope="Original isolated xq_rope execution. Identity-angle fixture tests explicit supplied column order, not a full model convention. DSD length mismatch behavior is recorded without assuming valid source semantics on arbitrary shape.",
            cases=reports,
            analyzer_sha256=sha(Path(__file__)),
        ),
        indent=2,
    )
    + "\n"
)
print(a.output)
print(reports)
