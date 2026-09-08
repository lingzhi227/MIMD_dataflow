"""Execute and audit triangular CSL factorization with warm reentry."""

import json
import os
from pathlib import Path
import subprocess


def read(root, name):
    return json.loads((Path(root) / name).read_text())


def run(root):
    from factor_sdk import run_factor

    return run_factor(root, "tile", "f_chol", {"P": "P", "Nt": "Nt"})


def check_factor(a, actual):
    """Independent LAPACK factor and reconstruction checks; no adaptive tolerance."""
    import numpy as np
    from frontend import check

    a, actual = np.asarray(a, np.float64), np.asarray(actual, np.float64)
    check(
        a.ndim == 2 and a.shape[0] == a.shape[1] and actual.shape == a.shape,
        "Cholesky shape",
    )
    check(np.all(np.isfinite(a)) and np.all(np.isfinite(actual)), "Cholesky finite")
    check(np.array_equal(a, a.T), "Cholesky symmetric input")
    expected = np.linalg.cholesky(a)
    check(
        np.array_equal(actual, np.tril(actual)), "Cholesky upper triangle must be zero"
    )
    check(np.all(np.diag(actual) > 0), "Cholesky positive diagonal")
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
    residual = float(
        np.linalg.norm(actual @ actual.T - a, ord=np.inf)
        / np.linalg.norm(a, ord=np.inf)
    )
    check(residual <= 3e-5, "Cholesky relative reconstruction residual")
    return {
        "contract": "fixed-cholesky-factor-and-LLT-v1",
        "fixed_accuracy_passed": True,
        "max_factor_abs_error": float(np.max(np.abs(actual - expected))),
        "relative_infinity_residual": residual,
        "residual_limit": 3e-5,
    }


def checkpoint_reference(a, p):
    """Independent float64 global right-looking state at the two tile corners."""
    import numpy as np

    a = np.tril(np.asarray(a, np.float64)).copy()
    n = a.shape[0]
    nt = n // p
    history = np.zeros((p, p, n, 2), np.float64)
    for k in range(n):
        a[k, k] = np.sqrt(a[k, k])
        a[k + 1 :, k] /= a[k, k]
        column = a[k + 1 :, k]
        a[k + 1 :, k + 1 :] -= np.tril(np.outer(column, column))
        for y in range(p):
            for x in range(y + 1):
                if k < (x + 1) * nt:
                    history[y, x, k] = (
                        a[y * nt, x * nt],
                        a[(y + 1) * nt - 1, (x + 1) * nt - 1],
                    )
    return history


def audit(root, manifest):
    import tempfile
    import numpy as np
    from frontend import check
    from float32 import close
    from ir import evaluate
    from mesh_cholesky import plan, generate

    root = Path(root)
    m, s, r = (
        read(root, name) for name in ("semantic.json", "schedule.json", "results.json")
    )
    batches = read(root, "batches.json")
    check(s == plan(m), "Cholesky schedule regeneration")
    with tempfile.TemporaryDirectory() as temp:
        generate(s, temp)
        for name in ("pe.csl", "layout.csl", "launch.csl"):
            check(
                (root / name).read_bytes() == (Path(temp) / name).read_bytes(),
                "Cholesky CSL regeneration",
            )
    check(
        r["success"] and len(r["cases"]) == len(r["diagnostics"]) == len(batches),
        "Cholesky incomplete",
    )
    check(
        r["runtime_instances"] == 1 and r["prepare_barriers"] == len(batches),
        "Cholesky lifecycle",
    )
    check(close(r["cases"], evaluate(m, batches)[0]), "Cholesky HLS/device mismatch")
    checks, cycles = [], []
    checkpoint_error = 0.0
    checkpoint_count = 0
    p, nt, n = s["P"], s["Nt"], s["N"]
    for batch, case, diag in zip(batches, r["cases"], r["diagnostics"]):
        checkpoint_expected = checkpoint_reference(
            np.asarray(batch[m["nodes"][0]["host"]]).reshape(n, n), p
        )
        checkpoint_actual = np.asarray(diag["checkpoints"])
        check(checkpoint_actual.shape == (p, p, n, 2), "Cholesky checkpoint shape")
        np.testing.assert_allclose(
            checkpoint_actual, checkpoint_expected, rtol=3e-5, atol=3e-6
        )
        checkpoint_error = max(
            checkpoint_error,
            float(np.max(np.abs(checkpoint_actual - checkpoint_expected))),
        )
        checkpoint_count += 2 * sum(
            (x + 1) * nt for y in range(p) for x in range(y + 1)
        )
        check(set(case) == {m["nodes"][2]["host"]}, "Cholesky output ports")
        checks.append(
            check_factor(
                np.asarray(batch[m["nodes"][0]["host"]]).reshape(n, n),
                np.asarray(case[m["nodes"][2]["host"]]).reshape(n, n),
            )
        )
        check(
            np.asarray(diag["timing"]).shape == (p, p, 6)
            and np.asarray(diag["progress"]).shape == (p, p, 2),
            "Cholesky diagnostics shape",
        )
        for y in range(p):
            for x in range(p):
                check(
                    diag["progress"][y][x] == ([(x + 1) * nt, 1] if x <= y else [0, 0]),
                    f"Cholesky progress p{x}_{y}",
                )
                words = diag["timing"][y][x]
                limit = (x + 1) * nt if x <= y else 0
                np.testing.assert_array_equal(
                    checkpoint_actual[y, x, limit:], np.zeros((n - limit, 2))
                )
                if x > y:
                    check(words == [0] * 6, "Cholesky inactive timing")
                    continue
                ticks = [
                    (sum(int(words[i + base]) << (16 * i) for i in range(3)))
                    for base in (0, 3)
                ]
                duration = (ticks[1] - ticks[0]) % (1 << 48)
                check(0 < duration < 1 << 32, "Cholesky timestamp bounds")
                cycles.append(duration)
    report = {
        "passed": True,
        "fixed_accuracy_passed": True,
        "actors": p * (p + 1) // 2,
        "epochs": len(batches),
        "output_values": manifest["expected_output_values"],
        "factor_checks": checks,
        "internal_observations": checkpoint_count,
        "max_checkpoint_abs_error": checkpoint_error,
        "checkpoint_scope": "Two tile corners after each active global pivot; inactive/unexecuted entries checked against zero",
        "per_pe_factor_cycles": cycles,
        "max_pe_factor_cycles": max(cycles),
        "timing_scope": "Per-PE SDK simulator factorization including communication; excludes host prepare and I/O. No hardware or speedup claim.",
        "source_sha256": manifest["source_sha256"],
    }
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
