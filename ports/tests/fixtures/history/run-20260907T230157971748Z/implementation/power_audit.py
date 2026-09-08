"""Original-operator replay and protocol audit for fixed-step resident power."""

import json, math
from pathlib import Path
import numpy as np
from frontend import check
from power_ir import inputs, solve
from solver_audit import load, apply, replicated, protocol


def audit(root):
    root = Path(root)
    m, s, r, batches = load(root)
    cap = s["max_iterations"]
    reports = []
    for epoch, (b, raw, d) in enumerate(zip(batches, r["cases"], r["diagnostics"])):
        a, initial, requested = inputs(m, b)
        o = {n["inputs"][0].split(".")[1]: raw[n["host"]] for n in m["nodes"][6:]}
        for field, key in [("reason", "cg_reason"), ("iterations", "cg_iterations")]:
            check(
                len(o[field]) == 1 and type(o[field][0]) is int, "power integer result"
            )
            np.testing.assert_array_equal(replicated(d, key, s, 1), o[field])
        history = replicated(d, "cg_history", s, cap + 1)
        inv = replicated(d, "power_inverse", s, cap)
        attempts = int(replicated(d, "power_attempts", s, 1)[0])
        np.testing.assert_array_equal(history[:cap], o["norms"])
        vector = np.asarray(o["vector"])
        check(
            vector.shape == (s["N"],) and np.all(np.isfinite(vector)),
            "power vector domain",
        )
        check(
            np.asarray(d["cg_solution"]).shape
            == (s["rows"], s["cols"], s["geometry"]["local_vec_sz"]),
            "power ownership",
        )
        np.testing.assert_array_equal(vector, np.asarray(d["cg_solution"]).ravel())
        reason, k = o["reason"][0], o["iterations"][0]
        check(reason in (0, 1) and 0 <= k <= requested, "qualified finite power status")
        check(
            attempts == k + (reason == 1) and attempts <= requested,
            "power attempt accounting",
        )
        check(reason != 0 or k == requested, "power fixed budget completion")
        np.testing.assert_array_equal(history[attempts:], 0)
        np.testing.assert_array_equal(inv[k:], 0)
        x = np.asarray(initial, dtype=float)
        last_input = x.copy()
        for i in range(attempts):
            last_input = x.copy()
            y = apply(a, x)
            nr = math.hypot(*y)
            np.testing.assert_allclose(history[i], nr, rtol=3e-5, atol=0)
            if i == k:
                check(reason == 1 and nr == 0, "power zero norm guard")
                break
            check(history[i] > 0, "power positive normalization")
            np.testing.assert_allclose(inv[i], 1.0 / history[i], rtol=3e-6, atol=0)
            x = np.asarray(y * inv[i], dtype=np.float32).astype(float)
        np.testing.assert_allclose(vector, x, rtol=3e-5, atol=3e-6)
        if k == 0:
            np.testing.assert_array_equal(vector, initial)
        else:
            np.testing.assert_allclose(math.hypot(*vector), 1.0, rtol=3e-6, atol=0)
        reference = solve(a, initial, requested, cap)
        check(
            reference["reason"] == [reason] and reference["iterations"] == [k],
            "power reference status",
        )
        np.testing.assert_allclose(vector, reference["vector"], rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(history[:cap], reference["norms"], rtol=3e-5, atol=0)
        cycles = protocol(s, a, last_input, d, epoch, attempts, 2 * attempts, k)
        ax = apply(a, vector)
        denom = float(np.dot(vector, vector))
        rayleigh = float(np.dot(vector, ax) / denom) if denom else None
        residual = math.hypot(*(ax - rayleigh * vector)) if denom else None
        reports.append(
            dict(
                reason=reason,
                completed_steps=k,
                attempts=attempts,
                operator_witness_active=bool(attempts),
                cycles=cycles,
                rayleigh_quotient=rayleigh,
                eigen_residual=residual,
                dominance_certified=False,
            )
        )
    report = dict(
        passed=True,
        profile=s["profile"],
        epochs=len(batches),
        cases=reports,
        contract="fixed steps; completion does not certify eigenpair convergence",
    )
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
