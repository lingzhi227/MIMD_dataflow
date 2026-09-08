"""Source-only normalized FFN/residual: exact target stages and original-input math."""

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
from mesh_rms import reference as rms_reference
from mesh_common import pack_tiles, unpack_tiles
from sdk_math_reference import silu_f16


def review(root, preflight=False):
    verify(root)
    provenance = read(root / "provenance.json")
    assert (
        provenance["normalized_feed_forward_residual"]
        and provenance["blocked_down_accumulation"]
        and provenance["preserve_completed_left_ownership"]
    )
    blocked_upper = provenance.get("blocked_upper_accumulation", False)
    geometry = read(root / "geometry.json")
    m, n, f, p = [geometry[k] for k in ("M", "N", "F", "P")]
    bs = read(root / "logical-inputs.json")
    assert len(bs) == 3
    results = None
    if not preflight:
        e = read(root / "execution.json")
        assert e["success"] and e["results_sha256"] == sha(root / "results.json")
        results = read(root / "results.json")
        assert (
            results["success"]
            and results["runtime_instances"] == 1
            and len(results["cases"]) == 3
        )
    q = lambda v: np.asarray(v, np.float16).astype(float)
    bit = lambda v: np.asarray(v, np.float16).view(np.uint16)
    pack = lambda v: pack_tiles(v, p, p, "F")
    rows = []
    for epoch, b in enumerate(bs):
        z = np.asarray(b["z"]).reshape(m, n)
        gamma = np.asarray(b["gamma"]).reshape(1, n)
        u = np.asarray(b["up_weight"]).reshape(n, f)
        g = np.asarray(b["gate_weight"]).reshape(n, f)
        d = np.asarray(b["down_weight"]).reshape(f, n)
        s = dict(M=m, N=n, Mt=m // p, Nt=n // p, rows=p, cols=p, epsilon=1e-6)
        *_, normalized = rms_reference(s, z, gamma)
        up, _, up_wide = projection(normalized, u, p, blocked=blocked_upper)
        gate, _, gate_wide = projection(normalized, g, p, blocked=blocked_upper)
        act = np.vectorize(silu_f16, otypes=[float])(gate)
        hidden = q(up * act)
        down, _, wide = projection(hidden, d, p, blocked=True)
        target = q(z + down)
        observed = target
        observed_down = down
        if results is not None:
            v = results["cases"][epoch]
            schema = read(root / "schema.json")
            assert set(v) == set(schema["outputs"])
            for key, length in schema["outputs"].items():
                raw = np.asarray(v[key])
                width = schema.get("output_word_bits", {}).get(key, 16)
                assert raw.shape == (p, p, length)
                assert np.issubdtype(raw.dtype, np.integer) and np.all(
                    (raw >= 0) & (raw < 2**width)
                )
            timestamps = np.asarray(v["timing"], np.int64)
            cycles = sum(
                (timestamps[:, :, j + 3] - timestamps[:, :, j]) * (1 << (16 * j))
                for j in range(3)
            ) % (1 << 48)
            assert np.all((cycles > 0) & (cycles < 2**32))

            for key, array in (
                ("normalized", normalized),
                ("up", up),
                ("activated_gate", act),
                ("output", down),
                ("result", target),
            ):
                np.testing.assert_array_equal(
                    np.asarray(v[key], np.uint16), bit(pack(array))
                )
            np.testing.assert_array_equal(
                np.asarray(v["wide_accumulator"], np.uint32), wide
            )
            if blocked_upper:
                for name, words in (("up", up_wide), ("gate", gate_wide)):
                    np.testing.assert_array_equal(
                        np.asarray(v[name + "_accumulator"], np.uint32), words
                    )
            np.testing.assert_array_equal(v["progress"], epoch + 1)
            observed = unpack_tiles(
                np.asarray(v["result"], np.uint16).view(np.float16), m // p, n // p, "F"
            ).astype(float)
            observed_down = unpack_tiles(
                np.asarray(v["output"], np.uint16).view(np.float16), m // p, n // p, "F"
            ).astype(float)
        mean = np.asarray(
            [math.fsum(float(x) * float(x) for x in row) / n for row in z]
        )
        norm_math = z * gamma / np.sqrt(mean[:, None] + 1e-6)
        up_math = standard_product(norm_math, u)
        gate_math = standard_product(norm_math, g)
        hidden_math = up_math * gate_math / (1 + np.exp(-gate_math))
        down_math = standard_product(hidden_math, d)
        nominal = z + down_math

        def accuracy(actual, expected):
            err = actual - expected
            l2 = float(np.linalg.norm(err)) / max(
                float(np.linalg.norm(expected)), 1e-30
            )
            peak = float(np.max(np.abs(err))) / max(
                float(np.max(np.abs(expected))), 1e-30
            )
            assert np.all(np.isfinite(actual)) and l2 <= 0.02 and peak <= 0.03, (
                l2,
                peak,
            )
            return dict(
                relative_l2=l2, peak_scaled_error=peak, fixed_accuracy_passed=True
            )

        rows.append(
            dict(
                epoch=epoch,
                final_residual=accuracy(observed, nominal),
                mlp_delta=accuracy(observed_down, down_math),
                target_bits_observed=results is not None,
            )
        )
    paths = [
        root / "provenance.json",
        Path(__file__),
        ROOT / "experiments/analyze_mlp_source.py",
        ROOT / "lib/Conversion/mesh_rms.py",
        ROOT / "lib/Numerics/sdk_math_reference.py",
    ]
    if results is not None:
        paths.append(root / "results.json")
    return dict(
        blocked_upper_accumulation=blocked_upper,
        passed=True,
        preflight_only=preflight,
        new_sdk_execution=False,
        cases=rows,
        scope="Supplied Z -> repaired source RMS library/row collective -> original MLP with gate-owner repair and explicit block-f32 down (and up/gate when blocked_upper_accumulation is true) -> original add_result updates Z. Shared target models versus separate original-input fsum/sqrt/exp math; delta checked separately so residual cannot conceal MLP errors. Source-only three calls, not HLS/full Prefill/Decode or hardware qualification.",
        hashes={str(v): sha(v) for v in paths},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("probe", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    assert not a.output.exists()
    report = review(a.probe, a.preflight)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(a.output, report["cases"])
