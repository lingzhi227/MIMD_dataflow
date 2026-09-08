"""Executed source h1/z_add/RMS/MLP/add_result: target bits and whole-input math."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify
from analyze_mlp_source import projection

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from attention_tail_fixtures import check
from mesh_score import reference as score_reference
from mesh_softmax import reference as softmax_reference
from mesh_device_matmul import reference as value_reference
from sdk_math_reference import sqrt_f16
import math
from mesh_rms import reference as rms_reference
from mesh_common import pack_tiles, unpack_tiles
from sdk_math_reference import silu_f16


def review(root, preflight=False):
    root = Path(root).resolve()
    verify(root)
    provenance = read(root / "provenance.json")
    assert all(
        provenance[k]
        for k in (
            "resident_supplied_qkv_tail",
            "normalized_feed_forward_residual",
            "blocked_upper_accumulation",
            "blocked_down_accumulation",
            "preserve_completed_left_ownership",
        )
    )
    g = read(root / "geometry.json")
    m, n, f, p = [g[k] for k in ("M", "N", "F", "P")]
    bs = read(root / "logical-inputs.json")
    assert len(bs) == 3
    result = None
    if not preflight:
        e = read(root / "execution.json")
        assert e["success"] and e["results_sha256"] == sha(root / "results.json")
        result = read(root / "results.json")
        assert (
            result["success"]
            and result["runtime_instances"] == 1
            and len(result["cases"]) == 3
        )
    q = lambda a: np.asarray(a, np.float16).astype(float)
    pack = lambda a: pack_tiles(a, p, p, "F")
    bits = lambda a: np.asarray(a, np.float16).view(np.uint16)
    rows = []
    for epoch, b in enumerate(bs):
        qq, kk, vv = [np.asarray(b[key]).reshape(m, n) for key in ("q", "k", "v")]
        geom = dict(P=p, Mt=m // p, Nt=n // p)
        _, _, _, logits = score_reference(geom, qq, kk)
        scale = float(np.float16(1 / sqrt_f16(n)))
        soft = dict(rows=p, cols=p, M=m, N=m, Mt=m // p, Nt=m // p, scale=scale)
        _, _, probability = softmax_reference(soft, logits)
        *_, a = value_reference(geom, probability, vv)
        o = np.asarray(b["output_weight"]).reshape(n, n)
        r = np.asarray(b["residual"]).reshape(m, n)
        gamma = np.asarray(b["gamma"]).reshape(1, n)
        u = np.asarray(b["up_weight"]).reshape(n, f)
        g = np.asarray(b["gate_weight"]).reshape(n, f)
        d = np.asarray(b["down_weight"]).reshape(f, n)
        projected, _, _ = projection(a, o, p)
        z = q(projected + r)
        s = dict(M=m, N=n, Mt=m // p, Nt=n // p, rows=p, cols=p, epsilon=1e-6)
        *_, norm = rms_reference(s, z, gamma)
        up, _, uw = projection(norm, u, p, blocked=True)
        gate, _, gw = projection(norm, g, p, blocked=True)
        act = np.vectorize(silu_f16, otypes=[float])(gate)
        hidden = q(up * act)
        delta, _, dw = projection(hidden, d, p, blocked=True)
        final = q(z + delta)
        observed_projection, observed_delta, observed_final = projected, delta, final
        observed_attention = a
        observed_score = logits
        observed_probability = probability
        if result is not None:
            v = result["cases"][epoch]
            schema = read(root / "schema.json")
            assert set(v) == set(schema["outputs"])
            for key, length in schema["outputs"].items():
                raw = np.asarray(v[key])
                width = schema.get("output_word_bits", {}).get(key, 16)
                assert (
                    raw.shape == (p, p, length)
                    and np.issubdtype(raw.dtype, np.integer)
                    and np.all((raw >= 0) & (raw < 2**width))
                )
            for key, array in (
                ("logits", logits),
                ("probability", probability),
                ("attention", a),
                ("projection", projected),
                ("post_projection_residual", z),
                ("normalized", norm),
                ("up", up),
                ("activated_gate", act),
                ("output", delta),
                ("result", final),
            ):
                np.testing.assert_array_equal(
                    np.asarray(v[key], np.uint16), bits(pack(array))
                )
            for key, array in (
                ("up_accumulator", uw),
                ("gate_accumulator", gw),
                ("wide_accumulator", dw),
            ):
                np.testing.assert_array_equal(np.asarray(v[key], np.uint32), array)
            np.testing.assert_array_equal(v["scale"], bits(np.full((p, p, 1), scale)))
            np.testing.assert_array_equal(v["progress"], epoch + 1)
            t = np.asarray(v["timing"], np.int64)
            cycles = sum(
                (t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)
            ) % (1 << 48)
            assert np.all((cycles > 0) & (cycles < 2**32))

            def decoded(key):
                return unpack_tiles(
                    np.asarray(v[key], np.uint16).view(np.float16), m // p, n // p, "F"
                ).astype(float)

            observed_attention = decoded("attention")
            observed_score = unpack_tiles(
                np.asarray(v["logits"], np.uint16).view(np.float16), m // p, m // p, "F"
            ).astype(float)
            observed_probability = unpack_tiles(
                np.asarray(v["probability"], np.uint16).view(np.float16),
                m // p,
                m // p,
                "F",
            ).astype(float)
            observed_projection = decoded("projection")
            observed_delta = decoded("output")
            observed_final = decoded("result")
        row = check(
            m,
            n,
            f,
            1e-6,
            1 / math.sqrt(n),
            b,
            {"output": observed_final.ravel().tolist()},
            observed_attention,
            observed_projection,
            observed_delta,
            score=observed_score,
            probability=observed_probability,
        )
        rows.append(
            dict(
                epoch=epoch,
                target_bits_observed=result is not None,
                original_input_math=row,
            )
        )
    paths = [
        root / "provenance.json",
        Path(__file__),
        ROOT / "tests/support/attention_tail_fixtures.py",
        ROOT / "tests/support/prefill_tail_fixtures.py",
        ROOT / "lib/Conversion/mesh_score.py",
        ROOT / "lib/Conversion/mesh_softmax.py",
        ROOT / "lib/Conversion/mesh_device_matmul.py",
        ROOT / "tests/support/feed_forward_fixtures.py",
        ROOT / "experiments/analyze_mlp_source.py",
        ROOT / "lib/Conversion/mesh_rms.py",
        ROOT / "lib/Numerics/sdk_math_reference.py",
        ROOT / "lib/Numerics/projection_reference.py",
    ]
    if result is not None:
        paths.append(root / "results.json")
    return dict(
        passed=True,
        preflight_only=preflight,
        new_sdk_execution=False,
        cases=rows,
        scope="Supplied Q/K/V, output weight, original residual, gamma and MLP weights. Original score/softmax/value/O/residual/RMS/MLP/final-Z chain with documented maximum, V layout, descriptor, RMS and gate-ownership repairs; explicit block-f32 up/gate/down. Original-nine-input fsum/sqrt/exp separately checks logits, probabilities and mass, attention output, projection, MLP delta and final output. Three source calls, single-head/unmasked; not HLS/full Prefill/Decode/QKV projections/RoPE/masks/cache/hardware qualification.",
        hashes={str(v.resolve().relative_to(ROOT)): sha(v) for v in paths},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("probe", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    assert not a.output.exists()
    r = review(a.probe, a.preflight)
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(a.output, r["cases"])
