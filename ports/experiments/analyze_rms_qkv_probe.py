"""Check original resident normalization fan-out and aligned-input reuse."""

import argparse, json, sys, math
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from mesh_normalized_matmul import reference
from mesh_common import unpack_tiles
from normalized_matmul_fixtures import check

p = argparse.ArgumentParser()
p.add_argument("probe", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
verify(a.probe)
e = read(a.probe / "execution.json")
assert e["success"] and e["results_sha256"] == sha(a.probe / "results.json")
s = read(a.probe / "schedule.json")
s["P"] = s["cols"]
r = read(a.probe / "results.json")
bs = read(a.probe / "inputs.json")
assert len(bs) == len(r["cases"]) == 2 and r["runtime_instances"] == 1
cases = []
for epoch, (b, c) in enumerate(zip(bs, r["cases"])):
    x = np.asarray(b["x"]).reshape(s["M"], s["N"])
    w = np.asarray(b["w"])
    reports = {}
    bits = lambda v: np.asarray(v, np.float16).view(np.uint16)
    for key, port in [("q", "hls_result"), ("k", "hls_k"), ("v", "hls_v")]:
        matrix = np.asarray(b[key]).reshape(s["N"], s["N"])
        norm, _, target = reference(s, x, w, matrix)
        raw = np.asarray(c[port])
        assert (
            raw.shape == (s["P"], s["P"], s["Mt"] * s["Nt"])
            and np.issubdtype(raw.dtype, np.integer)
            and np.all((raw >= 0) & (raw < 65536))
        )
        actual = unpack_tiles(
            raw.astype(np.uint16).view(np.float16), s["Mt"], s["Nt"], "F"
        ).astype(float)
        mismatches = int(np.count_nonzero(bits(actual) != bits(target)))
        np.testing.assert_array_equal(
            np.asarray(c["hls_normalized"], np.uint16),
            bits(__import__("mesh_common").pack_tiles(norm, s["P"], s["P"], "F")),
        )
        try:
            numeric = check(
                s["M"],
                s["N"],
                dict(x=b["x"], w=b["w"], q=b[key]),
                {"projected": actual.ravel().tolist()},
            )
            fixed = True
        except AssertionError as error:
            numeric = dict(error=str(error))
            fixed = False
        reports[key] = dict(
            target_half_mismatches=mismatches,
            standard=numeric,
            passed=mismatches == 0 and fixed,
        )
    np.testing.assert_array_equal(c["hls_progress"], epoch + 1)
    t = np.asarray(c["hls_time"], np.int64)
    cycles = sum((t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)) % (
        1 << 48
    )
    assert np.all((cycles > 0) & (cycles < 2**32))
    cases.append(dict(projections=reports, max_local_cycles=int(cycles.max())))
report = dict(
    passed=all(v["passed"] for c in cases for v in c["projections"].values()),
    scope="Original resident RMSNorm->Q/K/V continuation with repaired row scaling and feature-column host ownership. Original already-shifted input reuse preserved. Independent standard math.fsum composition and target scheduled half checks; no HLS or full inference qualification.",
    cases=cases,
    hashes={
        str(v): sha(v)
        for v in [
            a.probe / "provenance.json",
            a.probe / "results.json",
            ROOT / "normalized_matmul_fixtures.py",
            ROOT / "toolchain/mesh_normalized_matmul.py",
            ROOT / "toolchain/mesh_rms.py",
            ROOT / "toolchain/sdk_math_reference.py",
            Path(__file__),
        ]
    },
)
a.output.write_text(json.dumps(report, indent=2) + "\n")
print(a.output, report["passed"])
print(cases)
assert report["passed"]
