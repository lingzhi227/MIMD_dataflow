"""Executed source h1/z_add/RMS/MLP/add_result: target bits and whole-input math."""

import argparse, json, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify
from analyze_mlp_source import projection

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from prefill_tail_fixtures import check
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
            "supplied_attention_output_tail",
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
        a = np.asarray(b["attention"]).reshape(m, n)
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

            observed_projection = decoded("projection")
            observed_delta = decoded("output")
            observed_final = decoded("result")
        row = check(
            m,
            n,
            f,
            1e-6,
            b,
            {"output": observed_final.ravel().tolist()},
            observed_projection,
            observed_delta,
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
        ROOT / "prefill_tail_fixtures.py",
        ROOT / "feed_forward_fixtures.py",
        ROOT / "experiments/analyze_mlp_source.py",
        ROOT / "toolchain/mesh_rms.py",
        ROOT / "toolchain/sdk_math_reference.py",
        ROOT / "toolchain/projection_reference.py",
    ]
    if result is not None:
        paths.append(root / "results.json")
    return dict(
        passed=True,
        preflight_only=preflight,
        new_sdk_execution=False,
        cases=rows,
        scope="Supplied attention output, output weight, original residual, gamma and three MLP weights. Original h1_matmul/z_add, repaired RMS library and gate ownership, explicit block-f32 up/gate/down, original final add_result retaining postprojection Z. Projection and MLP delta checked separately against original-input fsum/sqrt/exp, besides final result. Three source calls; not HLS, full Prefill/Decode or hardware qualification.",
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
