"""FFN transport and actual normalized/delta/final observations around shared MLP audit."""

import json
from pathlib import Path
import numpy as np
from frontend import check
from mesh_common import pack_tiles, unpack_tiles
from mesh_feed_forward import core, plan, inputs, normalized
from mesh_mlp_sdk import (
    WIDE_PORTS,
    parameters,
    packed as mlp_packed,
    extents as mlp_extents,
    audit_cases as mlp_audit,
)
from half_region_runtime import read


def extents(s):
    return dict(
        mlp_extents(s),
        gamma=s["Nt"],
        normalized=s["length"],
        down_snapshot=s["length"],
        rms_progress=2,
    )


def packed(s, m, b):
    z, gamma, u, g, d = inputs(m, b)
    p = s["P"]
    return dict(
        mlp_packed(s, (z, u, g, d)),
        gamma=np.broadcast_to(gamma.reshape(1, p, s["Nt"]), (p, p, s["Nt"])),
    )


def decode(s, m, d):
    y = unpack_tiles(
        np.asarray(d["result"], np.uint16).view(np.float16), s["Mt"], s["Nt"], "F"
    ).astype(float)
    return {m["nodes"][-1]["host"]: y.ravel().tolist()}


def run(root):
    from half_region_runtime import run as execute

    execute(
        root,
        parameters,
        extents,
        packed,
        decode,
        word_bits={
            k: 32 for k in extents(read(root, "schedule.json")) if k in WIDE_PORTS
        },
    )


def accuracy(s, arrays, actual, delta):
    z, gamma, u, g, d = arrays
    x = z * gamma / np.sqrt(np.mean(z * z, axis=1)[:, None] + s["epsilon"])
    up = x @ u
    gate = x @ g
    expected_delta = (up * gate / (1 + np.exp(-gate))) @ d
    expected = z + expected_delta

    def compare(a, b):
        error = a - b
        l2 = float(np.linalg.norm(error)) / max(float(np.linalg.norm(b)), 1e-30)
        peak = float(np.max(np.abs(error))) / max(float(np.max(np.abs(b))), 1e-30)
        check(
            np.all(np.isfinite(a)) and l2 <= 0.02 and peak <= 0.03,
            "original-input FFN accuracy, including separate MLP delta",
        )
        return dict(relative_l2=l2, peak_scaled_error=peak, fixed_accuracy_passed=True)

    return dict(
        contract="normalized-feed-forward-half-normwise-v1",
        fixed_accuracy_passed=True,
        final_residual=compare(actual, expected),
        mlp_delta=compare(delta, expected_delta),
    )


def audit_cases(s, m, bs, r, require_complete=True):
    n = len(r["cases"])
    p = s["P"]
    c = core(m)
    check(
        len(r["diagnostics"]) == n and len(bs) == m["epochs"] and 1 <= n <= len(bs),
        "feed-forward completed calls",
    )
    bits = lambda a: np.asarray(a, np.float16).view(np.uint16)
    cb = []
    cd = []
    co = []
    numeric = []
    for epoch, b in enumerate(bs):
        arrays = inputs(m, b)
        z, gamma, u, g, d = arrays
        target_norm = normalized(s, z, gamma)
        cb.append(
            {
                node["host"]: a.ravel().tolist()
                for node, a in zip(c["nodes"][:4], (target_norm, u, g, d))
            }
        )
        if epoch >= n:
            continue
        observed = r["diagnostics"][epoch]
        check(set(observed) == set(extents(s)), "feed-forward observed ports")
        for key, length in extents(s).items():
            raw = np.asarray(observed[key])
            width = 32 if key in WIDE_PORTS else 16
            check(
                raw.shape == (p, p, length)
                and np.issubdtype(raw.dtype, np.integer)
                and np.all((raw >= 0) & (raw < 2**width)),
                "feed-forward observer shape/range",
            )
        for key, a in packed(s, m, b).items():
            np.testing.assert_array_equal(observed[key], bits(a))
        np.testing.assert_array_equal(
            observed["normalized"], bits(pack_tiles(target_norm, p, p, "F"))
        )
        np.testing.assert_array_equal(
            observed["rms_progress"], np.broadcast_to([1, epoch + 1], (p, p, 2))
        )
        shadow = {k: observed[k] for k in mlp_extents(s)}
        shadow["x"] = observed["normalized"]
        shadow["result"] = observed["down_snapshot"]
        cd.append(shadow)
        co.append(decode(s, c, shadow))
        actual = np.asarray(r["cases"][epoch][m["nodes"][-1]["host"]]).reshape(z.shape)
        check(
            set(r["cases"][epoch]) == {m["nodes"][-1]["host"]},
            "feed-forward output port",
        )
        np.testing.assert_array_equal(
            actual.ravel(), decode(s, m, observed)[m["nodes"][-1]["host"]]
        )
        delta = unpack_tiles(
            np.asarray(observed["down_snapshot"], np.uint16).view(np.float16),
            s["Mt"],
            s["Nt"],
            "F",
        ).astype(float)
        np.testing.assert_array_equal(
            np.asarray(observed["result"], np.uint16),
            bits(pack_tiles((z + delta).astype(np.float16), p, p, "F")),
        )
        numeric.append(accuracy(s, arrays, actual, delta))
    report = mlp_audit(s, c, cb, dict(r, diagnostics=cd, cases=co), require_complete)
    for row, number in zip(report["cases"], numeric):
        row["core_numerical"] = row.pop("numerical")
        row["numerical"] = number
        row["normalized_and_final_residual_bits_exact"] = True
    report["normalization_and_delta_half_observations"] = n * p * p * 2 * s["length"]
    report["performance_scope"] = (
        "Supplied Z -> RMS -> shared gated MLP -> Z residual, local WSE3 simulator intervals. Explicit block-f32 down and observed narrowed delta; no full Prefill/Decode or hardware throughput claim"
    )
    return report


def audit(root):
    from integrity import verify_bundle, verify_codegen

    root = Path(root)
    verify_bundle(root)
    verify_codegen(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    check(s == plan(m), "feed-forward schedule regeneration")
    result = audit_cases(s, m, read(root, "batches.json"), read(root, "results.json"))
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
