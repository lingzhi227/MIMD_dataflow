"""Projection/residual/RMS packing, completed-call diagnostics and independent audit."""

import json
from pathlib import Path
import numpy as np
from frontend import check
from mesh_common import pack_tiles, unpack_tiles
from mesh_twohop import block_index
from mesh_projection_residual_rms import plan, inputs, reference
from compiler_parameters import encode
from half_region_runtime import read


def parameters(s):
    p = s["P"]
    values = dict(
        P=p,
        dim_p_pe=s["Nt"],
        pes_p_head=p,
        pes_p_kv_head=p,
        head_dim_p_pe=s["Nt"],
        seq_len_p_pe=s["Mt"],
        ffn_dim_p_pe=s["Nt"],
        sampled=int(s["instrumentation"] == "sampled"),
        epsilon_bits=int(np.float16(s["epsilon"]).view(np.uint16)),
    )
    return encode({k: "u16" if k == "epsilon_bits" else "i16" for k in values}, values)


def packed(s, m, b):
    a, w, r, g = inputs(m, b)
    p = s["P"]
    nt = s["Nt"]
    return dict(
        activation=pack_tiles(a, p, p, "F"),
        residual=pack_tiles(r, p, p, "F"),
        gamma=np.broadcast_to(g.reshape(1, p, nt), (p, p, nt)),
        weight=np.asarray(
            [
                [
                    w[
                        block_index(p, y, x) * nt : (block_index(p, y, x) + 1) * nt,
                        x * nt : (x + 1) * nt,
                    ].ravel(order="C")
                    for x in range(p)
                ]
                for y in range(p)
            ]
        ),
    )


def extents(s):
    p, l, q, mt = [s[k] for k in ("P", "length", "weight_length", "Mt")]
    sample = s["instrumentation"] == "sampled"
    return dict(
        activation=l,
        weight=q,
        residual=l,
        gamma=s["Nt"],
        result=l,
        history=p * l if sample else 1,
        sum=l if sample else 1,
        left_first=l if sample else 1,
        right_first=q if sample else 1,
        local_square_sum=mt if sample else 1,
        reduced_square_sum=mt if sample else 1,
        inverse=mt,
        progress=4,
        timing=6,
        queues=2,
    )


def decode(s, m, d):
    y = unpack_tiles(
        np.asarray(d["result"], np.uint16).view(np.float16), s["Mt"], s["Nt"], "F"
    )
    return {m["nodes"][-1]["host"]: y.astype(float).ravel().tolist()}


def run(root):
    from half_region_runtime import run as execute

    execute(root, parameters, extents, packed, decode)


def accuracy(s, arrays, actual):
    a, w, r, g = arrays
    z = a @ w + r
    nominal = z * g / np.sqrt(np.mean(z * z, axis=1)[:, None] + s["epsilon"])
    e = actual - nominal
    l2 = float(np.linalg.norm(e)) / max(float(np.linalg.norm(nominal)), 1e-30)
    peak = float(np.max(np.abs(e))) / max(float(np.max(np.abs(nominal))), 1e-30)
    check(
        np.all(np.isfinite(actual)) and l2 <= 0.02 and peak <= 0.03,
        "original-input projection/residual/RMS accuracy",
    )
    return dict(
        contract="projection-residual-rms-half-normwise-v1",
        fixed_accuracy_passed=True,
        relative_l2=l2,
        peak_scaled_error=peak,
    )


def audit_cases(s, m, bs, r, require_complete=True):
    n = len(r["cases"])
    check(
        r["runtime_instances"] == 1
        and len(bs) == m["epochs"]
        and len(r["diagnostics"]) == n
        and r["launches"] == ["hls_main"] * n
        and 1 <= n <= len(bs),
        "region lifecycle",
    )
    complete = bool(r["success"] and n == len(bs))
    check(not r["success"] or complete, "no premature completion")
    if require_complete:
        check(complete, "complete SDK run required")
    sample = s["instrumentation"] == "sampled"
    p = s["P"]
    words = 0
    reports = []
    bits = lambda v: np.asarray(v, np.float16).view(np.uint16)
    for epoch, (b, o, d) in enumerate(zip(bs, r["cases"], r["diagnostics"])):
        arrays = inputs(m, b)
        target, witness = reference(s, *arrays)
        check(set(d) == set(extents(s)), "observed ports")
        raw = {}
        for key, count in extents(s).items():
            a = np.asarray(d[key])
            check(
                a.shape == (p, p, count)
                and np.issubdtype(a.dtype, np.integer)
                and np.all((a >= 0) & (a < 65536)),
                "raw half/u16 observer shape and range",
            )
            raw[key] = a.astype(np.uint16)
        for key, v in packed(s, m, b).items():
            np.testing.assert_array_equal(raw[key], bits(v))
        for key, v in witness.items():
            if sample or key == "inverse":
                np.testing.assert_array_equal(raw[key], bits(v))
                words += v.size
            else:
                check(np.all(raw[key] == 0), "inactive observer " + key)
        np.testing.assert_array_equal(
            raw["progress"], np.broadcast_to([1, p, 1, epoch + 1], (p, p, 4))
        )
        check(np.all((raw["queues"] & 248) == 248), "communication queues drained")
        actual = unpack_tiles(
            raw["result"].view(np.float16), s["Mt"], s["Nt"], "F"
        ).astype(float)
        np.testing.assert_array_equal(bits(actual), bits(target))
        check(set(o) == {m["nodes"][-1]["host"]}, "output host port")
        np.testing.assert_array_equal(
            np.asarray(o[m["nodes"][-1]["host"]]), actual.ravel()
        )
        numeric = accuracy(s, arrays, actual)
        t = raw["timing"].astype(np.int64)
        cycles = sum(
            (t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)
        ) % (1 << 48)
        check(np.all((cycles > 0) & (cycles < 2**32)), "valid local timestamps")
        reports.append(
            dict(
                target_half_bits_exact=True,
                numerical=numeric,
                max_local_cycles=int(cycles.max()),
                cycles_per_pe=cycles.tolist(),
            )
        )
    return dict(
        passed=complete,
        complete=complete,
        completed_calls_valid=True,
        profile=s["profile"],
        epochs=n,
        instrumentation=s["instrumentation"],
        internal_half_observations=words,
        cases=reports,
        performance_scope="Local WSE3 simulator intervals; supplied activation/weight/residual/gamma, no full model or hardware throughput claim",
    )


def audit(root):
    from integrity import verify_bundle, verify_codegen

    root = Path(root)
    verify_bundle(root)
    verify_codegen(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    check(s == plan(m), "schedule regeneration")
    report = audit_cases(s, m, read(root, "batches.json"), read(root, "results.json"))
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
