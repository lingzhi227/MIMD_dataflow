"""Target-stage/protocol audit for the resident mixed graph.

Uses original public inputs for the prefix and observed, checked producer joins
for each subsequent recurrence. Independent original-input application accuracy
is an additional qualification gate, never replaced by this audit.
"""

import copy, math
import numpy as np
from frontend import check
from mesh_input_attention import inputs
from mesh_input_attention_mixed_sdk import extents, packed, WIDE_PORTS
from mesh_common import unpack_tiles
from mesh_twohop import cycle
from mesh_rms import reference as rms_reference
from mesh_pair_rotation import reference as pair_reference
from mesh_score import reference as score_reference
from projection_reference import project
from mixed_matmul_reference import evaluate as wide_matmul
from mesh_mlp import verify_graph, plan as mlp_plan
from mesh_mlp_sdk import audit_cases as mlp_audit, extents as mlp_extents
from input_contracts import effective_bound


def mlp_module(m, s):
    ns = copy.deepcopy(m["nodes"])
    norm = ns[23]
    host = "__observed_normalized"
    while host in {n.get("host") for n in ns}:
        host += "_"
    norm = dict(
        id=norm["id"],
        op="input",
        inputs=[],
        host=host,
        shape=norm["shape"],
        dtype="f16",
        line=norm["line"],
        abs_bound=s["mlp_numerical_bounds"]["inputs"][0],
    )
    weights = ns[8:11]
    for n in weights:
        n["abs_bound"] = effective_bound(n, m["input_bound"])
    out = dict(ns[-1], inputs=[ns[28]["id"]])
    return verify_graph(
        dict(m, nodes=[norm, *weights, *ns[24:29], out]),
        m["epochs"],
        max(m["input_bound"], math.ceil(norm["abs_bound"])),
    )


def audit_cases(s, m, bs, r, require_complete=True):
    rows = r["diagnostics"]
    count = len(rows)
    check(
        r["runtime_instances"] == 1
        and len(bs) == m["epochs"]
        and len(r["cases"]) == count
        and r["launches"] == ["hls_main"] * count
        and 1 <= count <= m["epochs"],
        "mixed region lifecycle",
    )
    complete = bool(r["success"] and count == m["epochs"])
    check(
        not r["success"] or complete,
        "successful mixed region must complete every epoch",
    )
    if require_complete:
        check(complete, "complete mixed execution required")
    p = s["P"]
    ring = cycle(p)
    hbits = lambda a: np.asarray(a, np.float16).view(np.uint16)
    fbits = lambda a: np.asarray(a, np.float32).view(np.uint32)
    mm = mlp_module(m, s)
    ms = mlp_plan(mm)
    md = []
    mb = []
    mo = []
    reports = []
    for epoch, (batch, row, out) in enumerate(zip(bs, rows, r["cases"])):
        x, gamma, qw, kw, vw, c, sine, o, u, g, d = inputs(m, batch)
        check(set(row) == set(extents(s)), "mixed diagnostic ports")
        for name, n in extents(s).items():
            raw = np.asarray(row[name])
            width = 32 if name in WIDE_PORTS else 16
            check(
                raw.shape == (p, p, n)
                and np.issubdtype(raw.dtype, np.integer)
                and np.all((raw >= 0) & (raw < 2**width)),
                "mixed raw word extent/range " + name,
            )
        for name, a in packed(s, m, batch).items():
            np.testing.assert_array_equal(
                row[name], hbits(a), err_msg="immutable " + name
            )

        def value(name):
            wide = name in WIDE_PORTS
            a = unpack_tiles(
                np.asarray(row[name], np.uint32 if wide else np.uint16).view(
                    np.float32 if wide else np.float16
                ),
                s["Mt"],
                (
                    s["Mt"]
                    if name in ("attention_logits", "mixed_probability_snapshot")
                    else s["Nt"]
                ),
                "F",
            ).astype(float)
            check(np.all(np.isfinite(a)), "finite mixed observation " + name)
            return a

        normal = rms_reference(
            dict(
                rows=p,
                cols=p,
                M=s["M"],
                N=s["N"],
                Mt=s["Mt"],
                Nt=s["Nt"],
                epsilon=s["epsilon"],
            ),
            x,
            gamma,
        )[-1]
        np.testing.assert_array_equal(hbits(value("input_normalized")), hbits(normal))
        for stage, weight, raw, pair in (
            (0, qw, "input_q_raw", "x"),
            (1, kw, "input_k_raw", "attention_k"),
        ):
            projected = project(s["input_prefix_stages"][stage], normal, weight)[0]
            np.testing.assert_array_equal(hbits(value(raw)), hbits(projected))
            np.testing.assert_array_equal(
                hbits(value(pair)),
                hbits(pair_reference(value(raw), c, sine, "odd_even", True)[-1]),
            )
        score = score_reference(s, value("x"), value("attention_k"))[-1]
        np.testing.assert_array_equal(hbits(value("attention_logits")), hbits(score))
        for left, right, name in (
            (normal, vw, "mixed_v"),
            (value("mixed_probability_snapshot"), value("mixed_v"), "mixed_a"),
            (value("mixed_a"), o, "mixed_projection"),
        ):
            np.testing.assert_array_equal(
                fbits(value(name)),
                fbits(wide_matmul(left, right, p)),
                err_msg="f32 recurrence " + name,
            )
        prob = value("mixed_probability_snapshot")
        exp = np.exp((score - score.max(axis=1)[:, None]) * s["scale"])
        expected = exp / exp.sum(axis=1)[:, None]
        mass = float(np.max(np.abs(prob.sum(axis=1) - 1)))
        prel = float(np.max(np.abs(prob / expected - 1)))
        check(mass <= 2e-6 and prel <= 2e-6, "observed-score f32 softmax accuracy")
        z = value("mixed_z")
        np.testing.assert_array_equal(
            fbits(z), fbits(np.float32(value("mixed_projection")) + np.float32(x))
        )
        norm = value("mixed_normalized")
        nominal = z * gamma / np.sqrt(np.mean(z * z, axis=1)[:, None] + s["epsilon"])
        error = float(
            np.linalg.norm(norm - nominal) / max(np.linalg.norm(nominal), 1e-30)
        )
        check(error <= 2e-6, "observed-Z f32 RMS accuracy")
        np.testing.assert_array_equal(hbits(value("normalized")), hbits(norm))
        np.testing.assert_array_equal(
            hbits(value("result")),
            hbits(np.float32(z) + np.float32(value("down_snapshot"))),
        )
        check(set(out) == {m["nodes"][-1]["host"]}, "mixed final output port")
        np.testing.assert_array_equal(
            out[m["nodes"][-1]["host"]], value("result").ravel()
        )
        for y in range(p):
            for col in range(p):
                offset = (-ring.index(y)) % p
                for name, wanted in {
                    "input_prefix_progress": [
                        1,
                        offset,
                        p,
                        p,
                        p,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        epoch + 1,
                    ],
                    "prelude_progress": [offset, p, 1, epoch + 1],
                    "attention_progress": [
                        (-ring.index(col)) % p,
                        offset,
                        p,
                        epoch + 1,
                    ],
                    "score_progress": [p, 1, epoch + 1],
                    "score_roots": [ring[(ring.index(y) - k) % p] for k in range(p)],
                    "rms_progress": [1, epoch + 1],
                    "attention_softmax_progress": [1] * 6,
                }.items():
                    np.testing.assert_array_equal(
                        row[name][y][col], wanted, err_msg=name
                    )
        diag = {name: row[name] for name in mlp_extents(ms)}
        diag = dict(diag, x=row["normalized"], result=row["down_snapshot"])
        md.append(diag)
        mb.append(
            {
                mm["nodes"][0]["host"]: value("normalized").ravel().tolist(),
                **{n["host"]: batch[n["host"]] for n in mm["nodes"][1:4]},
            }
        )
        mo.append({mm["nodes"][-1]["host"]: value("down_snapshot").ravel().tolist()})
        reports.append(
            dict(
                epoch=epoch,
                half_prefix_exact=True,
                f32_contractions_exact=True,
                residual_and_narrowing_exact=True,
                probability_mass_error=mass,
                probability_max_relative_error=prel,
                rms_relative_l2=error,
            )
        )
    mlp = mlp_audit(
        ms,
        mm,
        mb if complete else [*mb, *({} for _ in range(m["epochs"] - count))],
        dict(
            success=complete,
            runtime_instances=1,
            launches=r["launches"],
            diagnostics=md,
            cases=mo,
        ),
        require_complete=require_complete,
    )
    return dict(
        passed=complete,
        complete=complete,
        completed_calls_valid=True,
        epochs=count,
        cases=reports,
        mlp=mlp,
        f32_contraction_words=count * 3 * s["M"] * s["N"],
        scope=__doc__,
    )
