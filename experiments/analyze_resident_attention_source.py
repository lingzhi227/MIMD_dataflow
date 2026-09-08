"""Resident single-head attention source audit, with independent original-input reference."""

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
from mesh_device_matmul import reference as value_reference
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
sampled = read(a.probe / "provenance.json")["instrumentation"] == "sampled"
reports = []
for epoch, (b, c) in enumerate(zip(bs, r["cases"])):
    q = unpack_tiles(np.asarray(b["q"]), mt, nt, "F")
    k = unpack_tiles(np.asarray(b["k"]), mt, nt, "F")
    partial, owners, roots, logits = score_reference(s, q, k)
    history, exponents, target = softmax_reference(soft, logits)
    np.testing.assert_array_equal(c["scale"], bits(np.full((P, P, 1), scale)))
    if sampled:
        np.testing.assert_array_equal(c["logits"], bits(pack_tiles(logits, P, P, "F")))
        np.testing.assert_array_equal(c["history"], bits(partial))
        np.testing.assert_array_equal(
            c["probability"], bits(pack_tiles(target, P, P, "F"))
        )
    else:
        for key in ("logits", "history", "probability", "value_history"):
            np.testing.assert_array_equal(c[key], np.zeros((P, P, 1), np.uint16))
    value_input = unpack_tiles(np.asarray(b["v"]), mt, nt, "F")
    vh, left, right, output = value_reference(s, target, value_input)
    if sampled:
        np.testing.assert_array_equal(c["value_history"], bits(vh))
    np.testing.assert_array_equal(c["output"], bits(pack_tiles(output, P, P, "F")))
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
    nominal_output = np.asarray(
        [
            [
                math.fsum(float(x) * float(y) for x, y in zip(row, col))
                for col in value_input.T
            ]
            for row in nominal
        ]
    )
    output_error = output - nominal_output
    output_l2 = float(np.linalg.norm(output_error)) / max(
        float(np.linalg.norm(nominal_output)), 1e-30
    )
    output_peak = float(np.max(np.abs(output_error))) / max(
        float(np.max(np.abs(nominal_output))), 1e-30
    )
    assert np.all(np.isfinite(output)) and output_l2 <= 0.02 and output_peak <= 0.025
    reports.append(
        dict(
            target_output_bits_exact=True,
            internal_half_bits_exact=True if sampled else None,
            internal_tensors_observed=sampled,
            probability_metrics_source=(
                "observed exact probability"
                if sampled
                else "target reference model only; counter has no probability observation"
            ),
            output_relative_l2=output_l2,
            output_peak_scaled_error=output_peak,
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
    scope="Source-only resident unmasked single-head softmax(QK-transpose times scale) times V with maximum-only repair, device V alignment/strided DSD adapter and contiguous score DSD reset on reentry. No intermediate host transfer. Original-input math.fsum dots, math.exp/fsum normalization and final math.fsum products, separate from target-half trajectory. Not HLS/full model qualification.",
    hashes={
        str(x): sha(x)
        for x in [
            a.probe / "provenance.json",
            a.probe / "results.json",
            Path(__file__),
            ROOT / "lib/Conversion/mesh_score.py",
            ROOT / "lib/Conversion/mesh_device_matmul.py",
            ROOT / "lib/Conversion/mesh_softmax.py",
            ROOT / "lib/Numerics/sdk_math_reference.py",
            ROOT / "lib/Numerics/binary16.py",
            ROOT / "lib/Conversion/mesh_twohop.py",
            ROOT / "lib/Conversion/mesh_common.py",
        ]
    },
)
a.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
