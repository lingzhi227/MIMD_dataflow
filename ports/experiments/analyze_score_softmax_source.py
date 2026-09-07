"""Resident score/softmax source audit, with independent original-input reference."""

import argparse, json, math, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from mesh_score import reference as score_reference
from mesh_softmax import reference as softmax_reference
from mesh_common import unpack_tiles, pack_tiles
from sdk_math_reference import sqrt_f16

p = argparse.ArgumentParser()
p.add_argument("probe", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
verify(a.probe)
e = read(a.probe / "execution.json")
assert e["success"] and e["results_sha256"] == sha(a.probe / "results.json")
r = read(a.probe / "results.json")
bs = read(a.probe / "inputs.json")
g = read(a.probe / "geometry.json")
M, N, P = g["M"], g["N"], g["P"]
mt, nt = M // P, N // P
assert r["success"] and r["runtime_instances"] == 1 and len(bs) == len(r["cases"]) == 3
bits = lambda v: np.asarray(v, np.float16).view(np.uint16)
scale = float(np.float16(1 / sqrt_f16(N)))
s = dict(P=P, Mt=mt, Nt=nt)
soft = dict(rows=P, cols=P, M=M, N=M, Mt=mt, Nt=mt, scale=scale)
reports = []
for epoch, (b, c) in enumerate(zip(bs, r["cases"])):
    q = unpack_tiles(np.asarray(b["q"]), mt, nt, "F")
    k = unpack_tiles(np.asarray(b["k"]), mt, nt, "F")
    partial, owners, roots, logits = score_reference(s, q, k)
    history, exponents, target = softmax_reference(soft, logits)
    np.testing.assert_array_equal(c["scale"], bits(np.full((P, P, 1), scale)))
    np.testing.assert_array_equal(c["logits"], bits(pack_tiles(logits, P, P, "F")))
    np.testing.assert_array_equal(c["history"], bits(partial))
    np.testing.assert_array_equal(c["score"], bits(pack_tiles(target, P, P, "F")))
    np.testing.assert_array_equal(c["peak"], bits(history[:, :, 1]))
    np.testing.assert_array_equal(c["inverse"], bits(history[:, :, 4]))
    np.testing.assert_array_equal(c["progress"], epoch + 1)
    standard_logits = (
        np.array(
            [
                [
                    math.fsum(float(x) * float(y) for x, y in zip(qrow, krow))
                    for krow in k
                ]
                for qrow in q
            ]
        )
        * scale
    )
    nominal = []
    for row in standard_logits:
        v = [math.exp(float(x) - float(max(row))) for x in row]
        total = math.fsum(v)
        nominal.append([x / total for x in v])
    nominal = np.asarray(nominal)
    err = target - nominal
    l2 = float(np.linalg.norm(err)) / float(np.linalg.norm(nominal))
    peak = float(np.max(np.abs(err))) / float(np.max(np.abs(nominal)))
    mass = float(np.max(np.abs(target.sum(axis=1) - 1)))
    assert l2 <= 0.015 and peak <= 0.02 and mass <= 0.01
    reports.append(
        dict(
            target_score_and_probability_bits_exact=True,
            standard_relative_l2=l2,
            standard_peak_scaled_error=peak,
            max_row_mass_error=mass,
        )
    )
report = dict(
    passed=True,
    cases=reports,
    scale=scale,
    new_sdk_execution=False,
    scope="Source-only resident score to repaired stable softmax, no intermediate host transfer. Original-input math.fsum dots and math.exp/fsum normalization, separate from target-half trajectory. Not HLS/full attention qualification.",
    hashes={
        str(x): sha(x)
        for x in [
            a.probe / "provenance.json",
            a.probe / "results.json",
            Path(__file__),
            ROOT / "toolchain/mesh_score.py",
            ROOT / "toolchain/mesh_softmax.py",
            ROOT / "toolchain/sdk_math_reference.py",
        ]
    },
)
a.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
