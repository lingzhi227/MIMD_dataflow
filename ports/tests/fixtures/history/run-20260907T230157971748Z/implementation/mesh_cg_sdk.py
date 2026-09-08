"""One-launch resident CG SDK binding; original-CSC post-execution audit."""

import json, math, tempfile
from pathlib import Path
from mesh_spmv_sdk import read


def run(root):
    from resident_sdk import run as resident_run

    return resident_run(root)


def audit(root, manifest):
    import numpy as np
    from frontend import check
    from mesh_cg import plan, generate, packing, TEMPLATES
    from solver_reference import matrix_inputs, solve, diagonal

    root = Path(root)
    if read(root, "schedule.json").get("solver") == "bicgstab":
        from bicgstab_sdk import audit as bicgstab_audit

        return bicgstab_audit(root, manifest)
    m = read(root, "semantic.json")
    s = read(root, "schedule.json")
    r = read(root, "results.json")
    batches = read(root, "batches.json")
    from resident_abi import schema

    check(read(root, "host-abi.json") == schema(s), "CG host ABI regeneration")
    jacobi = s.get("preconditioner") == "jacobi"
    check(s == plan(m), "CG schedule regeneration")
    with tempfile.TemporaryDirectory() as tmp:
        generate(s, tmp)
        for name in TEMPLATES:
            check(
                (root / name).read_bytes() == (Path(tmp) / name).read_bytes(),
                "CG CSL regeneration " + name,
            )
    check(
        read(root, "sparse-packing.json") == packing(m, batches),
        "CG matrix packing regeneration",
    )
    check(
        r["success"]
        and r["runtime_instances"] == 1
        and len(r["cases"]) == len(r["diagnostics"]) == len(batches) == m["epochs"]
        and r["launches"] == ["f_cg"] * len(batches),
        "CG lifecycle",
    )
    reports = []
    cycles = []
    for epoch, (b, o, d) in enumerate(zip(batches, r["cases"], r["diagnostics"])):
        o = {n["inputs"][0].split(".")[1]: o[n["host"]] for n in m["nodes"][8:]}
        a, rhs, initial, limit, tols = matrix_inputs(m, b)

        def apply(x):
            terms = [[] for _ in x]
            for row, col, value in a.entries():
                terms[row].append(float(value) * float(x[col]))
            return np.array([math.fsum(v) for v in terms])

        x = np.array(o["solution"])
        k = o["iterations"][0]
        reason = o["reason"][0]
        history = np.array(o["residual_squared"])
        scalars = np.array(d["cg_scalars"][0][0]).reshape(-1, 3)
        check(
            type(k) is int
            and type(reason) is int
            and 0 <= k <= limit
            and reason in range(5),
            "CG integer result",
        )
        for name in (
            "cg_reason",
            "cg_iterations",
            "cg_history",
            "cg_true_norm",
            "cg_scalars",
            "cg_progress",
        ):
            data = np.array(d[name])
            np.testing.assert_array_equal(data, np.broadcast_to(data[0, 0], data.shape))
        for port, name in [
            ("reason", "cg_reason"),
            ("iterations", "cg_iterations"),
            ("residual_squared", "cg_history"),
            ("true_residual_norm", "cg_true_norm"),
        ]:
            check(o[port] == d[name][0][0], "CG structured output transport")
        np.testing.assert_array_equal(x, np.array(d["cg_solution"]).ravel())
        check(
            np.all(np.isfinite(x))
            and np.all(np.isfinite(history))
            and np.all(history >= 0),
            "CG finite result",
        )
        np.testing.assert_array_equal(
            history[k + 1 :], 0, err_msg="CG unused residual history tail"
        )
        np.testing.assert_array_equal(scalars[k:], 0)
        # Independent operator, reconstructed after execution using exported recurrence scalars.
        rr = np.asarray(np.array(rhs) - apply(initial), dtype=np.float32)
        p = rr.copy()
        xx = np.array(initial, dtype=np.float32)
        if jacobi:
            inv = np.asarray(1.0 / np.asarray(diagonal(a)), dtype=np.float32)
            np.testing.assert_allclose(
                np.asarray(d["cg_diagonal"]).ravel(), inv, rtol=3e-6, atol=0
            )
            weights = np.asarray(d["cg_weights"][0][0])
            np.testing.assert_array_equal(
                np.asarray(d["cg_weights"]),
                np.broadcast_to(weights, np.asarray(d["cg_weights"]).shape),
            )
            np.testing.assert_array_equal(weights[k + (1 if reason == 2 else 0) :], 0)
            p = np.asarray(rr * inv, dtype=np.float32)
        rho0 = float(np.dot(rr.astype(float), rr.astype(float)))
        initial_square_underflow = (
            0 < rho0 < float(np.nextafter(np.float32(0), np.float32(1))) / 2
        )
        if initial_square_underflow:
            check(
                reason == 3 and k == 0 and history[0] == 0,
                "CG initial squared-norm underflow classification",
            )
            np.testing.assert_array_equal(x, initial)
        else:
            np.testing.assert_allclose(history[0], rho0, rtol=3e-5, atol=0)
        for j in range(k):
            alpha, curvature, beta = scalars[j]
            if jacobi:
                z = np.asarray(rr * inv, dtype=np.float32)
                np.testing.assert_allclose(
                    weights[j],
                    float(np.dot(rr.astype(float), z.astype(float))),
                    rtol=1e-3,
                    atol=1e-10 * rho0,
                )
            ap = apply(p)
            check(
                curvature > 0 and alpha > 0 and np.isfinite(alpha),
                "CG positive finite step",
            )
            np.testing.assert_allclose(
                curvature,
                float(np.dot(p.astype(float), ap)),
                rtol=3e-4,
                atol=1e-10 * rho0,
            )
            np.testing.assert_allclose(
                alpha,
                np.float32((weights[j] if jacobi else history[j]) / curvature),
                rtol=3e-6,
                atol=0,
            )
            xx = np.asarray(
                xx.astype(float) + alpha * p.astype(float), dtype=np.float32
            )
            rr = np.asarray(rr.astype(float) - alpha * ap, dtype=np.float32)
            np.testing.assert_allclose(
                history[j + 1],
                float(np.dot(rr.astype(float), rr.astype(float))),
                rtol=1e-3,
                atol=1e-10 * rho0,
            )
            if j + 1 < k:
                np.testing.assert_allclose(
                    beta,
                    np.float32(
                        (weights[j + 1] / weights[j])
                        if jacobi
                        else (history[j + 1] / history[j])
                    ),
                    rtol=3e-6,
                    atol=0,
                )
                p = np.asarray(
                    (np.asarray(rr * inv, dtype=np.float32) if jacobi else rr).astype(
                        float
                    )
                    + beta * p.astype(float),
                    dtype=np.float32,
                )
        np.testing.assert_allclose(x, xx, rtol=3e-5, atol=3e-6)
        true = float(np.linalg.norm(np.array(rhs) - apply(x)))
        bnorm = float(np.linalg.norm(rhs))
        threshold = max(tols[0] * bnorm, tols[1])
        np.testing.assert_allclose(
            np.array(d["final_ax"]).ravel(), apply(x), rtol=3e-5, atol=3e-6
        )
        if initial_square_underflow:
            check(
                true > 0 and o["true_residual_norm"][0] > 0,
                "CG underflow retains positive stable residual",
            )
        np.testing.assert_allclose(
            o["true_residual_norm"][0], true, rtol=1e-3, atol=2e-7 * bnorm
        )
        if reason == 0:
            check(
                true <= 1.05 * threshold + 1e-7 * bnorm,
                "CG original system convergence",
            )
        # Original sequential C++/IR is an additional oracle, not device scheduling.
        ref = solve(a, rhs, initial, limit, tols, s["max_iterations"], jacobi=jacobi)
        check(
            o["reason"] == ref["reason"],
            "CG fixture-scoped reference reason (iteration count is reported separately)",
        )
        native_solution_screen = bool(
            np.allclose(x, ref["solution"], rtol=3e-5, atol=3e-6)
        )
        native_true = float(np.linalg.norm(np.asarray(rhs) - apply(ref["solution"])))
        if ref["reason"] == [0]:
            check(
                native_true <= 1.05 * threshold + 1e-7 * bnorm,
                "CG native original residual contract",
            )
        # First/last occupied partition rows before horizontal reduction.
        terms = [[{} for _ in range(s["cols"])] for _ in range(s["rows"])]
        g = s["geometry"]
        for row, col, value in a.entries():
            terms[row // g["block_rows"]][col // g["block_cols"]].setdefault(
                row, []
            ).append(float(value) * float(x[col]))
        for py in range(s["rows"]):
            for px in range(s["cols"]):
                occupied = sorted(terms[py][px])
                wanted = (
                    [
                        math.fsum(terms[py][px][row])
                        for row in (occupied[0], occupied[-1])
                    ]
                    if occupied
                    else [0.0, 0.0]
                )
                np.testing.assert_allclose(
                    d["partial"][py][px], wanted, rtol=3e-5, atol=3e-6
                )
        progress = np.array(d["progress"])
        np.testing.assert_array_equal(progress[:, :, :10], 0)
        np.testing.assert_array_equal(progress[:, :, 10], epoch + 1)
        cp = np.array(d["cg_progress"])[0, 0]
        calls = k + 2 + (1 if reason == 2 else 0)
        check(
            cp[0] == cp[1] == calls
            and cp[3] == k
            and cp[4] == epoch + 1
            and cp[5] == 1 + 3 * calls,
            "CG callback/phase counts",
        )
        expected_collectives = 7 + 2 * k + (1 if reason == 2 else 0)
        if jacobi:
            expected_collectives += k + (1 if reason == 2 else 0)
        if k > 0 and history[k] == 0:
            expected_collectives += 2
        check(
            cp[2] == expected_collectives, "CG fixture-scoped scalar collective count"
        )
        for mask in np.array(d["cg_queue_last"]).ravel():
            check(mask & 252 == 252, "CG drained resource boundary")
        ts = np.array(d["cg_timing"], dtype=np.int64)
        ticks = sum((ts[:, :, i + 3] - ts[:, :, i]) * (1 << (16 * i)) for i in range(3))
        check(np.all(ticks > 0) and np.all(ticks < 2**32), "CG timestamp interval")
        cycles.append(int(ticks.max()))
        reports.append(
            dict(
                reason=reason,
                iterations=k,
                native_iterations=ref["iterations"][0],
                native_solution_fixed_screen_passed=native_solution_screen,
                native_original_residual=native_true,
                native_strict_requested_tolerance_met=(native_true <= threshold),
                iteration_count_agrees=(k == ref["iterations"][0]),
                true_residual=true,
                requested_threshold=threshold,
                strict_requested_tolerance_met=(true <= threshold),
                rounding_allowance=0.05 * threshold + 1e-7 * bnorm,
                original_residual_within_rounding_allowance=(
                    true <= 1.05 * threshold + 1e-7 * bnorm
                ),
                spmv_calls=int(cp[0]),
                scalar_collectives=int(cp[2]),
            )
        )
    report = dict(
        passed=True,
        actors=s["rows"] * s["cols"],
        epochs=len(batches),
        output_values=manifest["expected_output_values"],
        solver_checks=reports,
        max_local_cycles=cycles,
        timing_scope="Maximum local interval of one resident solver launch, excludes host copies; simulator only, not synchronized global latency",
        source_sha256=manifest["source_sha256"],
    )
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
