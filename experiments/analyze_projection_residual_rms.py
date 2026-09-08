"""Separate source descriptor/index behavior from original-input composed math."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, sys, math
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify
from analyze_mlp_source import projection, standard_product

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from mesh_common import pack_tiles, unpack_tiles
from mesh_rms import reference as rms_reference
from sdk_math_reference import rms_inverse_f16


def analyze(root):
    verify(root)
    execution = read(root / "execution.json")
    assert execution["success"] and execution["results_sha256"] == sha(
        root / "results.json"
    )
    results = read(root / "results.json")
    bs = read(root / "logical-inputs.json")
    g = read(root / "geometry.json")
    assert (
        results["success"]
        and results["runtime_instances"] == 1
        and len(results["cases"]) == len(bs) == 3
    )
    m, n, p = [g[k] for k in ("M", "N", "P")]
    mt = m // p
    nt = n // p
    repair = read(root / "provenance.json")["repair"]
    q = lambda v: np.asarray(v, np.float16).astype(float)
    bits = lambda v: np.asarray(v, np.float16).view(np.uint16)
    rows = []
    for epoch, (b, d) in enumerate(zip(bs, results["cases"])):
        a = np.asarray(b["activation"]).reshape(m, n)
        w = np.asarray(b["weight"]).reshape(n, n)
        residual = np.asarray(b["residual"]).reshape(m, n)
        gamma = np.asarray(b["gamma"]).reshape(1, n)
        h, _, _ = projection(a, w, p)
        z = q(h + residual)
        s = dict(M=m, N=n, rows=p, cols=p, Mt=mt, Nt=nt, epsilon=1e-6)
        local, total, inv, normalized = rms_reference(s, z, gamma)
        if repair == "none":
            local = np.zeros_like(local)
            total = np.zeros_like(total)
            inv = np.full_like(inv, rms_inverse_f16(0, n, 1e-6))
        weighted = pack_tiles(q(z * gamma), p, p, "F").reshape(p, p, nt, mt)
        if repair not in ("both", "library"):
            # Source inverse[feature] scales an entire contiguous row vector.
            target = q(weighted * inv[:, :, :, None]).reshape(p, p, -1)
        else:
            target = pack_tiles(normalized, p, p, "F")
        expected = dict(
            projection=pack_tiles(h, p, p, "F"),
            sum=pack_tiles(z, p, p, "F"),
            result=target,
            local_square_sum=local,
            reduced_square_sum=total,
            inverse=inv,
        )
        mismatches = {
            k: int(np.count_nonzero(np.asarray(d[k]) != bits(v)))
            for k, v in expected.items()
        }
        assert not any(mismatches.values()), mismatches
        np.testing.assert_array_equal(d["progress"], epoch + 1)
        actual = unpack_tiles(
            np.asarray(d["result"], np.uint16).view(np.float16), mt, nt, "F"
        ).astype(float)
        nominal_z = standard_product(a, w) + residual
        mean_square = np.asarray(
            [math.fsum(float(v) * float(v) for v in row) / n for row in nominal_z]
        )
        nominal = nominal_z * gamma / np.sqrt(mean_square[:, None] + 1e-6)
        error = actual - nominal
        l2 = float(np.linalg.norm(error)) / max(float(np.linalg.norm(nominal)), 1e-30)
        peak = float(np.max(np.abs(error))) / max(float(np.max(np.abs(nominal))), 1e-30)
        rows.append(
            dict(
                epoch=epoch,
                source_behavior_bits_exact=True,
                original_input_relative_l2=l2,
                original_input_peak_scaled_error=peak,
                mathematical_accuracy_passed=bool(
                    np.all(np.isfinite(actual)) and l2 <= 0.02 and peak <= 0.03
                ),
            )
        )
    report = dict(
        observed_source_behavior_confirmed=True,
        mathematical_accuracy_passed=all(
            v["mathematical_accuracy_passed"] for v in rows
        ),
        repair=repair,
        cases=rows,
        new_sdk_execution=False,
        scope="Source projection/residual/RMS with explicit valid zero stale-descriptor seed. Source behavior and original-input fsum projection/residual/RMS are checked separately. Not an unmodified full Prefill or HLS qualification.",
        hashes={
            str(v): sha(v)
            for v in (
                root / "provenance.json",
                root / "results.json",
                Path(__file__),
                ROOT / "experiments/analyze_mlp_source.py",
                ROOT / "lib/Conversion/mesh_rms.py",
                ROOT / "lib/Numerics/sdk_math_reference.py",
            )
        },
    )
    if repair in ("both", "library"):
        assert report["mathematical_accuracy_passed"]
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("probe", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    assert not a.output.exists()
    r = analyze(a.probe)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(a.output, r["mathematical_accuracy_passed"])
