"""Original-CSC trajectory and termination audit for resident BiCGStab."""

import json, math
from pathlib import Path
import numpy as np
from frontend import check
from solver_reference import matrix_inputs
from bicgstab_reference import solve
from solver_audit import load, record, replicated, apply, protocol


def audit(root, manifest):
    root = Path(root)
    m, s, result, batches = load(root)
    cap = s["max_iterations"]
    reports = []
    cycles = []
    for epoch, (b, raw, d) in enumerate(
        zip(batches, result["cases"], result["diagnostics"])
    ):
        a, rhs, initial, limit, tols = matrix_inputs(m, b)
        o = record(m, s, raw, d)
        x = np.asarray(o["solution"])
        k = o["iterations"][0]
        reason = o["reason"][0]
        check(0 <= k <= limit and reason in (0, 1, 3, 4), "BiCGStab result domain")
        history = replicated(d, "cg_history", s, cap + 1)
        scalars = replicated(d, "cg_scalars", s, 3 * cap).reshape(cap, 3)
        products = replicated(d, "bi_products", s, 3 * cap).reshape(cap, 3)
        rhos = replicated(d, "bi_rhos", s, cap)
        ssaved = replicated(d, "bi_ss", s, cap)
        bp = replicated(d, "bi_progress", s, 9).astype(int)
        failure = int(replicated(d, "bi_failure", s, 1)[0])
        early = int(replicated(d, "bi_early", s, 1)[0])
        check(early in (0, 1), "BiCGStab early flag")
        np.testing.assert_array_equal(history[k + 1 :], 0)
        np.testing.assert_array_equal(rhos[bp[0] :], 0)
        np.testing.assert_array_equal(products[bp[0] :], 0)
        np.testing.assert_array_equal(ssaved[bp[3] :], 0)
        np.testing.assert_array_equal(scalars[bp[0] :], 0)
        rr = np.asarray(np.asarray(rhs) - apply(a, initial), dtype=np.float32)
        shadow = rr.copy()
        p = rr.copy()
        xx = np.asarray(initial, dtype=np.float32)
        rho0 = float(np.dot(rr.astype(float), rr.astype(float)))
        bnorm = float(np.linalg.norm(rhs))
        threshold = max(tols[0] * bnorm, tols[1])
        underflow = 0 < rho0 < float(np.nextafter(np.float32(0), np.float32(1))) / 2
        if underflow:
            check(
                reason == 3 and k == 0 and history[0] == 0 and failure == 0,
                "BiCGStab initial square underflow",
            )
            np.testing.assert_array_equal(x, initial)
        else:
            np.testing.assert_allclose(history[0], rho0, rtol=3e-5, atol=0)
        for j in range(k):
            alpha, omega, beta = scalars[j]
            denom, ts, tt = products[j]
            v = np.asarray(apply(a, p), dtype=np.float32)
            np.testing.assert_allclose(
                rhos[j],
                float(np.dot(shadow.astype(float), rr.astype(float))),
                rtol=1e-3,
                atol=1e-10 * rho0,
            )
            np.testing.assert_allclose(
                denom,
                float(np.dot(shadow.astype(float), v.astype(float))),
                rtol=1e-3,
                atol=1e-10 * rho0,
            )
            check(denom != 0 and rhos[j] != 0, "BiCGStab nonzero step denominator")
            np.testing.assert_allclose(
                alpha, np.float32(rhos[j] / denom), rtol=3e-6, atol=0
            )
            sv = np.asarray(
                rr.astype(float) - alpha * v.astype(float), dtype=np.float32
            )
            np.testing.assert_allclose(
                ssaved[j],
                float(np.dot(sv.astype(float), sv.astype(float))),
                rtol=1e-3,
                atol=1e-10 * rho0,
            )
            xx = np.asarray(
                xx.astype(float) + alpha * p.astype(float), dtype=np.float32
            )
            if early and j == k - 1:
                check(
                    reason in (0, 4)
                    and omega == 0
                    and ts == 0
                    and tt == 0
                    and beta == 0,
                    "BiCGStab early-s record",
                )
                check(
                    float(np.linalg.norm(sv)) <= 1.05 * threshold + 1e-7 * bnorm,
                    "BiCGStab early-s convergence",
                )
                rr = sv
            else:
                tv = np.asarray(apply(a, sv), dtype=np.float32)
                np.testing.assert_allclose(
                    [ts, tt],
                    [
                        float(np.dot(tv.astype(float), sv.astype(float))),
                        float(np.dot(tv.astype(float), tv.astype(float))),
                    ],
                    rtol=1e-3,
                    atol=1e-10 * rho0,
                )
                check(tt > 0 and omega != 0, "BiCGStab stabilization denominator")
                np.testing.assert_allclose(
                    omega, np.float32(ts / tt), rtol=3e-6, atol=0
                )
                xx = np.asarray(
                    xx.astype(float) + omega * sv.astype(float), dtype=np.float32
                )
                rr = np.asarray(
                    sv.astype(float) - omega * tv.astype(float), dtype=np.float32
                )
            np.testing.assert_allclose(
                history[j + 1],
                float(np.dot(rr.astype(float), rr.astype(float))),
                rtol=2e-3,
                atol=1e-10 * rho0,
            )
            if j + 1 < k:
                expected = np.float32(
                    np.float32(rhos[j + 1] / rhos[j]) * np.float32(alpha / omega)
                )
                np.testing.assert_allclose(beta, expected, rtol=3e-6, atol=0)
                p = np.asarray(
                    p.astype(float) - omega * v.astype(float), dtype=np.float32
                )
                p = np.asarray(
                    rr.astype(float) + beta * p.astype(float), dtype=np.float32
                )
        np.testing.assert_allclose(x, xx, rtol=3e-5, atol=3e-6)
        true = float(np.linalg.norm(np.asarray(rhs) - apply(a, x)))
        check(math.isfinite(o["true_residual_norm"][0]), "BiCGStab finite true norm")
        np.testing.assert_allclose(
            o["true_residual_norm"][0], true, rtol=1e-3, atol=2e-7 * bnorm
        )
        if reason == 0:
            check(
                true <= 1.05 * threshold + 1e-7 * bnorm,
                "BiCGStab original residual convergence",
            )
        if reason == 1:
            check(k == limit, "BiCGStab iteration budget")
        if reason == 3:
            if failure == 21:
                check(
                    k == 0 and not any(a.values) and products[0, 0] == 0,
                    "BiCGStab qualified zero-operator denominator failure",
                )
                np.testing.assert_array_equal(x, initial)
                np.testing.assert_array_equal(bp, [1, 0, 1, 0, 0, 0, 0, 0, 0])
            elif failure == 25:
                check(
                    k == 0 and products[0, 1] == 0 and products[0, 2] > 0,
                    "BiCGStab zero omega witness",
                )
                v = apply(a, rr)
                denom = float(np.dot(shadow.astype(float), v))
                check(
                    denom != 0,
                    "BiCGStab nonzero alpha denominator before omega failure",
                )
                alpha = rho0 / denom
                sv = rr - alpha * v
                tv = apply(a, sv)
                check(
                    float(np.dot(tv, sv)) == 0 and float(np.dot(tv, tv)) > 0,
                    "BiCGStab independent zero omega",
                )
                np.testing.assert_allclose(
                    products[0], [denom, 0.0, float(np.dot(tv, tv))], rtol=3e-5, atol=0
                )
                np.testing.assert_array_equal(x, initial)
                np.testing.assert_array_equal(bp, [1, 1, 1, 1, 1, 1, 0, 0, 0])
            else:
                check(
                    underflow
                    and failure == 0
                    and k == 0
                    and true > 0
                    and o["true_residual_norm"][0] > 0,
                    "BiCGStab qualified initial underflow",
                )
                np.testing.assert_array_equal(bp, 0)
        else:
            check(failure == 0, "BiCGStab no numerical failure")
            ap = k
            ass = k - early
            expected = [ap, ass, ap, ap, ass, ass, ass, max(0, k - 1), bp[8]]
            np.testing.assert_array_equal(bp, expected)
            expected_extra = int(early and k > 0 and ssaved[k - 1] == 0) + int(
                not early and k > 0 and history[k] == 0
            )
            check(bp[8] == expected_extra, "BiCGStab stable-norm fallback count")
        scalar_calls = 7 + int(sum(bp[2:8])) + 2 * int(bp[8])
        spmv_calls = 2 + int(bp[0] + bp[1])
        cycles.append(protocol(s, a, x, d, epoch, spmv_calls, scalar_calls, k))
        ref = solve(a, rhs, initial, limit, tols, cap)
        native_true = float(np.linalg.norm(np.asarray(rhs) - apply(a, ref["solution"])))
        check(o["reason"] == ref["reason"], "BiCGStab fixture-scoped native reason")
        if ref["reason"] == [0]:
            check(
                native_true <= 1.05 * threshold + 1e-7 * bnorm,
                "BiCGStab native original residual",
            )
        reports.append(
            dict(
                reason=reason,
                iterations=k,
                native_iterations=ref["iterations"][0],
                native_solution_fixed_screen_passed=bool(
                    np.allclose(x, ref["solution"], rtol=3e-5, atol=3e-6)
                ),
                native_original_residual=native_true,
                strict_requested_tolerance_met=true <= threshold,
                true_residual=true,
                requested_threshold=threshold,
                rounding_allowance=0.05 * threshold + 1e-7 * bnorm,
                spmv_calls=spmv_calls,
                scalar_collectives=scalar_calls,
                early_s=early,
                failure_stage=failure,
            )
        )
    report = dict(
        passed=True,
        actors=s["rows"] * s["cols"],
        epochs=len(batches),
        solver_checks=reports,
        max_local_cycles=cycles,
        output_values=manifest["expected_output_values"],
        source_sha256=manifest["source_sha256"],
        timing_scope="Maximum local resident solver interval; excludes H2D/D2H; simulator only, not global latency or hardware performance",
        breakdown_scope="Qualified failure trajectories: initial squared-norm underflow, zero-operator denominator and controlled zero omega; other finite/nonfinite breakdown paths remain unqualified",
    )
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
