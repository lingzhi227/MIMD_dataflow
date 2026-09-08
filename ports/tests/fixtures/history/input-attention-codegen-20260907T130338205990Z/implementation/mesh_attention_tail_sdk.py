"""Composed resident audits with immutable nine-input transport and observed joins."""

import json
from pathlib import Path
import numpy as np
from frontend import check
from mesh_attention_tail import attention, tail, inputs, plan
from mesh_attention import reference as attention_reference
from mesh_attention_sdk import (
    audit_cases as attention_audit,
    extents as attention_extents,
)
from mesh_prefill_tail_sdk import (
    extents as tail_extents,
    packed as tail_packed,
    audit_cases as tail_audit,
    accuracy as tail_accuracy,
)
from mesh_mlp_sdk import parameters, WIDE_PORTS
from mesh_feed_forward_sdk import decode
from mesh_common import pack_tiles, unpack_tiles
from half_region_runtime import read

ATTENTION_PORTS = {
    "q": "x",
    "k": "attention_k",
    "v": "attention_v",
    "result": "attention_snapshot",
    "logits": "attention_logits",
    "probability": "attention_probability",
    "history": "score_history",
    "owners": "score_owners",
    "roots": "score_roots",
    "progress": "score_progress",
    "value_history": "attention_value_history",
    "value_left": "attention_value_left",
    "value_right": "attention_value_right",
    "value_progress": "attention_progress",
    "exponents": "attention_exponents",
    "softmax_history": "attention_softmax_history",
    "softmax_progress": "attention_softmax_progress",
    "timing": "timing",
    "queues": "queues",
}


def extents(s):
    result = tail_extents(s)
    result.update(
        {
            ATTENTION_PORTS[k]: n
            for k, n in attention_extents(s["attention_schedule"]).items()
            if k not in ("q", "timing", "queues")
        }
    )
    result["attention_logits"] = result["attention_probability"] = s["score_length"]
    return result


def packed(s, m, b):
    arrays = inputs(m, b)
    tm = tail(m)
    # Transport physical x as Q. The synthetic A port is only a child packing view.
    tb = {
        v["host"]: a.ravel().tolist()
        for v, a in zip(tm["nodes"][:7], (arrays[0], *arrays[3:]))
    }
    result = tail_packed(s, tm, tb)
    result.update(
        attention_k=pack_tiles(arrays[1], s["P"], s["P"], "F"),
        attention_v=pack_tiles(arrays[2], s["P"], s["P"], "F"),
    )
    return result


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


def audit_cases(s, m, bs, r, require_complete=True):
    am, tm = attention(m), tail(m)
    acases, adiagnostics, ab, tb, td, numeric = [], [], [], [], [], []
    count = len(r["cases"])
    check(
        len(r["diagnostics"]) == count and 1 <= count <= len(bs),
        "attention-tail completed calls",
    )
    p = s["P"]
    bits = lambda a: np.asarray(a, np.float16).view(np.uint16)
    for epoch, b in enumerate(bs):
        arrays = inputs(m, b)
        q, k, v, *rest = arrays
        ab.append(
            {
                node["host"]: a.ravel().tolist()
                for node, a in zip(am["nodes"][:3], (q, k, v))
            }
        )
        target = attention_reference(s["attention_schedule"], q, k, v)
        logits, prob, a = target[3], target[6], target[-1]
        tb.append(
            {
                node["host"]: value.ravel().tolist()
                for node, value in zip(tm["nodes"][:7], (a, *rest))
            }
        )
        if epoch >= count:
            continue
        d = r["diagnostics"][epoch]
        check(set(d) == set(extents(s)), "attention-tail observed ports")
        for name, length in extents(s).items():
            raw = np.asarray(d[name])
            width = 32 if name in WIDE_PORTS else 16
            check(
                raw.shape == (p, p, length)
                and np.issubdtype(raw.dtype, np.integer)
                and np.all((raw >= 0) & (raw < 2**width)),
                "attention-tail raw shape/range: " + name,
            )
        for name, value in packed(s, m, b).items():
            np.testing.assert_array_equal(d[name], bits(value))
        for name, value in (
            ("attention_logits", logits),
            ("attention_probability", prob),
            ("attention_snapshot", a),
        ):
            np.testing.assert_array_equal(d[name], bits(pack_tiles(value, p, p, "F")))
        ad = {name: d[port] for name, port in ATTENTION_PORTS.items()}
        if s["instrumentation"] != "sampled":
            for name in ("logits", "probability"):
                ad[name] = np.zeros((p, p, 1), np.uint16).tolist()
        adiagnostics.append(ad)
        acases.append({am["nodes"][-1]["host"]: a.ravel().tolist()})
        shadow = {name: d[name] for name in tail_extents(s)}
        shadow["x"] = d["attention_snapshot"]
        td.append(shadow)
        # Entire chain uses original Q/K/V, without rounding the mathematical A.
        score = q @ k.T * s["scale"]
        e = np.exp(score - np.max(score, axis=1, keepdims=True))
        exact_attention = (e / np.sum(e, axis=1, keepdims=True)) @ v

        def decoded(name):
            return unpack_tiles(
                np.asarray(d[name], np.uint16).view(np.float16), s["Mt"], s["Nt"], "F"
            ).astype(float)

        numeric.append(
            tail_accuracy(
                s,
                (exact_attention, *rest),
                decoded("projection_snapshot"),
                decoded("down_snapshot"),
                decoded("result"),
            )
        )
    ar = attention_audit(
        s["attention_schedule"],
        am,
        ab,
        dict(r, cases=acases, diagnostics=adiagnostics),
        require_complete,
    )
    tr = tail_audit(s, tm, tb, dict(r, diagnostics=td), require_complete)
    for row, a, num in zip(tr["cases"], ar["cases"], numeric):
        row["observed_attention_tail_numerical"] = row.pop("numerical")
        row["numerical"] = num
        row["attention_numerical"] = a["numerical"]
        row["attention_bits_and_protocol_exact"] = True
    tr["attention_internal_half_observations"] = ar["internal_half_observations"]
    tr["internal_half_observations"] += ar["internal_half_observations"]
    tr["attention_mandatory_half_observations"] = (
        count * p * p * (2 * s["score_length"] + s["length"])
    )
    tr["performance_scope"] = (
        "Whole supplied-Q/K/V single-head unmasked attention/output/RMS/MLP/final-residual resident chain; local WSE3 simulator intervals, not full Prefill/Decode or hardware throughput."
    )
    return tr


def audit(root):
    from integrity import verify_bundle, verify_codegen

    root = Path(root)
    verify_bundle(root)
    verify_codegen(root)
    s, m = read(root, "schedule.json"), read(root, "semantic.json")
    check(s == plan(m), "attention-tail schedule regeneration")
    result = audit_cases(s, m, read(root, "batches.json"), read(root, "results.json"))
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
