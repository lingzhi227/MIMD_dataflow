"""Whole-input audit around observed projection/Z and the shared FFN SDK path."""

import json
from pathlib import Path
import numpy as np
from frontend import check
from mesh_prefill_tail import core, inputs, plan
from mesh_feed_forward_sdk import extents as ff_extents, audit_cases as ff_audit, decode
from mesh_mlp_sdk import packed as mlp_packed, parameters, WIDE_PORTS
from mesh_common import pack_tiles, unpack_tiles
from mesh_twohop import cycle
from projection_reference import project
from half_region_runtime import read


def extents(s):
    p, l, w = s["P"], s["length"], s["output_weight_length"]
    sample = s["instrumentation"] == "sampled"
    return dict(
        ff_extents(s),
        output_weight=w,
        residual=l,
        projection_snapshot=l,
        post_projection_z=l,
        projection_history=p * l if sample else 1,
        projection_left_first=l if sample else 1,
        projection_right_first=w if sample else 1,
        prelude_progress=4,
    )


def packed(s, m, b):
    a, o, r, gamma, u, g, d = inputs(m, b)
    p = s["P"]
    return dict(
        mlp_packed(s, (a, u, g, d)),
        output_weight=mlp_packed(s, (a, o, o, o))["up_weight"],
        residual=pack_tiles(r, p, p, "F"),
        gamma=np.broadcast_to(gamma.reshape(1, p, s["Nt"]), (p, p, s["Nt"])),
    )


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


def accuracy(s, arrays, observed_projection, observed_delta, observed_final):
    a, o, r, gamma, u, g, d = arrays
    projection = a @ o
    z = projection + r
    x = z * gamma / np.sqrt(np.mean(z * z, axis=1)[:, None] + s["epsilon"])
    up = x @ u
    gate = x @ g
    delta = (up * gate / (1 + np.exp(-gate))) @ d
    final = z + delta

    def compare(actual, expected):
        error = actual - expected
        l2 = float(np.linalg.norm(error)) / max(float(np.linalg.norm(expected)), 1e-30)
        peak = float(np.max(np.abs(error))) / max(
            float(np.max(np.abs(expected))), 1e-30
        )
        check(
            np.all(np.isfinite(actual)) and l2 <= 0.02 and peak <= 0.03,
            "whole original-input tail accuracy: separate projection/delta/final",
        )
        return dict(relative_l2=l2, peak_scaled_error=peak, fixed_accuracy_passed=True)

    return dict(
        contract="supplied-attention-tail-half-normwise-v1",
        fixed_accuracy_passed=True,
        projection=compare(observed_projection, projection),
        mlp_delta=compare(observed_delta, delta),
        final_residual=compare(observed_final, final),
    )


def audit_cases(s, m, bs, r, require_complete=True):
    n = len(r["cases"])
    p = s["P"]
    ring = cycle(p)
    c = core(m)
    check(
        len(bs) == m["epochs"] and len(r["diagnostics"]) == n and 1 <= n <= len(bs),
        "tail completed calls",
    )
    bits = lambda a: np.asarray(a, np.float16).view(np.uint16)
    cb = []
    cd = []
    numeric = []
    for epoch, b in enumerate(bs):
        arrays = inputs(m, b)
        a, o, residual, gamma, u, g, d = arrays
        projected, history, left, right, _ = project(s["projection_prelude"], a, o)
        z = (projected + residual).astype(np.float16).astype(float)
        cb.append(
            {
                node["host"]: v.ravel().tolist()
                for node, v in zip(c["nodes"][:5], (z, gamma, u, g, d))
            }
        )
        if epoch >= n:
            continue
        observed = r["diagnostics"][epoch]
        check(set(observed) == set(extents(s)), "tail observed ports")
        for key, length in extents(s).items():
            raw = np.asarray(observed[key])
            width = 32 if key in WIDE_PORTS else 16
            check(
                raw.shape == (p, p, length)
                and np.issubdtype(raw.dtype, np.integer)
                and np.all((raw >= 0) & (raw < 2**width)),
                "tail observer shape/range",
            )
        for key, v in packed(s, m, b).items():
            np.testing.assert_array_equal(observed[key], bits(v))
        for key, v in (("projection_snapshot", projected), ("post_projection_z", z)):
            np.testing.assert_array_equal(observed[key], bits(pack_tiles(v, p, p, "F")))
        for y in range(p):
            for x in range(p):
                np.testing.assert_array_equal(
                    observed["prelude_progress"][y][x],
                    [(-ring.index(y)) % p, p, 1, epoch + 1],
                )
        for key, v in (
            ("projection_history", history),
            ("projection_left_first", left),
            ("projection_right_first", right),
        ):
            if s["instrumentation"] == "sampled":
                np.testing.assert_array_equal(observed[key], bits(v))
            else:
                check(
                    np.all(np.asarray(observed[key]) == 0), "inactive prelude observer"
                )
        shadow = {k: observed[k] for k in ff_extents(s)}
        shadow["x"] = observed["post_projection_z"]
        cd.append(shadow)

        def decoded(key):
            return unpack_tiles(
                np.asarray(observed[key], np.uint16).view(np.float16),
                s["Mt"],
                s["Nt"],
                "F",
            ).astype(float)

        numeric.append(
            accuracy(
                s,
                arrays,
                decoded("projection_snapshot"),
                decoded("down_snapshot"),
                decoded("result"),
            )
        )
    report = ff_audit(s, c, cb, dict(r, diagnostics=cd), require_complete)
    for row, number in zip(report["cases"], numeric):
        row["ffn_numerical"] = row.pop("numerical")
        row["numerical"] = number
        row["projection_and_live_Z_bits_exact"] = True
    count = (
        n * p * p * (p * s["length"] + s["length"] + s["output_weight_length"])
        if s["instrumentation"] == "sampled"
        else 0
    )
    report["prelude_internal_half_observations"] = count
    report["internal_half_observations"] += count
    report["projection_and_live_Z_half_observations"] = n * p * p * 2 * s["length"]
    report["performance_scope"] = (
        "Whole supplied attention-output projection/residual/RMS/MLP/final-Z tail, local WSE3 simulator intervals. Not full attention/Prefill/Decode or hardware throughput."
    )
    return report


def audit(root):
    from integrity import verify_bundle, verify_codegen

    root = Path(root)
    verify_bundle(root)
    verify_codegen(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    check(s == plan(m), "tail schedule regeneration")
    r = audit_cases(s, m, read(root, "batches.json"), read(root, "results.json"))
    (root / "audit.json").write_text(json.dumps(r, indent=2) + "\n")
    return r
