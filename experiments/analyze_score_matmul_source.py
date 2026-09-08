"""Audit source QK transpose partials and measure root reduction order explicitly."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, math, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from binary16 import matmul
from mesh_common import unpack_tiles
from mesh_twohop import cycle

p = argparse.ArgumentParser()
p.add_argument("probe", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
verify(a.probe)
e = read(a.probe / "execution.json")
assert e["success"] and e["results_sha256"] == sha(a.probe / "results.json")
g = read(a.probe / "geometry.json")
m, n, P = g["M"], g["N"], g["P"]
mt, nt = m // P, n // P
order = cycle(P)
r = read(a.probe / "results.json")
bs = read(a.probe / "inputs.json")
assert r["success"] and r["runtime_instances"] == 1 and len(bs) == len(r["cases"]) == 3
bits = lambda x: np.asarray(x, np.float16).view(np.uint16)
half = lambda x: np.asarray(x, np.float16).astype(float)
reports = []
for epoch, (b, c) in enumerate(zip(bs, r["cases"])):
    qt = np.asarray(b["q"])
    kt = np.asarray(b["k"])
    partial = np.zeros((P, P, P, mt, mt))
    mismatches = 0
    for y in range(P):
        for x in range(P):
            left = qt[y, x].reshape(mt, nt, order="F")
            for step in range(P):
                owner = order[(order.index(y) - step) % P]
                right = kt[owner, x].reshape(mt, nt, order="F")
                partial[y, x, step] = matmul(left, right.T)
                observed = np.asarray(
                    c["history"][y][x][step * mt * mt : (step + 1) * mt * mt], np.uint16
                ).reshape(mt, mt, order="F")
                mismatches += int(
                    np.count_nonzero(observed != bits(partial[y, x, step]))
                )
    candidates = {}
    for east_first in (True, False):
        out = np.zeros((P, P, mt, mt))
        for y in range(P):
            for step in range(P):
                root = order[(order.index(y) - step) % P]
                v = partial[y, :, step]
                west = v[0].copy()
                for x in range(1, root):
                    west = half(west + v[x])
                east = v[-1].copy()
                for x in range(P - 2, root, -1):
                    east = half(east + v[x])
                total = v[root].copy()
                parts = ([east] if root < P - 1 else []) + ([west] if root > 0 else [])
                if not east_first:
                    parts.reverse()
                for part in parts:
                    total = half(total + part)
                out[y, root] = total
        actual = np.asarray(c["score"], np.uint16)
        expected = np.stack(
            [
                np.stack([bits(out[y, x]).ravel(order="F") for x in range(P)])
                for y in range(P)
            ]
        )
        candidates["east_first" if east_first else "west_first"] = int(
            np.count_nonzero(actual != expected)
        )
    Q = unpack_tiles(qt, mt, nt, "F")
    K = unpack_tiles(kt, mt, nt, "F")
    actual = unpack_tiles(
        np.asarray(c["score"], np.uint16).view(np.float16), mt, mt, "F"
    ).astype(float)
    standard = np.array(
        [
            [math.fsum(float(u) * float(v) for u, v in zip(qrow, krow)) for krow in K]
            for qrow in Q
        ]
    )
    error = actual - standard
    l2 = float(np.linalg.norm(error)) / max(float(np.linalg.norm(standard)), 1e-30)
    peak = float(np.max(np.abs(error))) / max(float(np.max(np.abs(standard))), 1e-30)
    np.testing.assert_array_equal(c["progress"], epoch + 1)
    t = np.asarray(c["timing"], np.int64)
    cycles = sum((t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)) % (
        1 << 48
    )
    reports.append(
        dict(
            epoch=epoch,
            partial_half_mismatches=mismatches,
            root_reduction_candidate_mismatches=candidates,
            standard_relative_l2=l2,
            standard_peak_scaled_error=peak,
            max_local_cycles=int(cycles.max()),
            passed=mismatches == 0
            and min(candidates.values()) == 0
            and l2 <= 0.015
            and peak <= 0.02,
        )
    )
report = dict(
    passed=all(x["passed"] for x in reports),
    cases=reports,
    new_sdk_execution=False,
    scope="Original score-only source schedule probe, not HLS or full attention qualification. Candidate reduction orders explicitly compared to actual device words.",
    hashes={
        str(x): sha(x)
        for x in [
            a.probe / "provenance.json",
            a.probe / "results.json",
            Path(__file__),
            ROOT / "lib/Numerics/binary16.py",
            ROOT / "lib/Conversion/mesh_common.py",
            ROOT / "lib/Conversion/mesh_twohop.py",
        ]
    },
)
a.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
