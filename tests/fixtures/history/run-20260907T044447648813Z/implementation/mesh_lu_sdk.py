"""SDK host binding and independent packed-LU/reconstruction audit."""

import json
import math
from pathlib import Path


def read(root, name):
    return json.loads((Path(root) / name).read_text())


def run(root):
    from factor_sdk import run_factor

    return run_factor(root, "A", "start", {"M": "N", "grid_size": "P"})


def check_factor(a, actual):
    import numpy as np
    from frontend import check

    a, actual = np.asarray(a, np.float64), np.asarray(actual, np.float64)
    check(
        a.ndim == 2 and a.shape[0] == a.shape[1] and actual.shape == a.shape, "LU shape"
    )
    check(np.all(np.isfinite(a)) and np.all(np.isfinite(actual)), "LU finite")
    n = a.shape[0]
    # Independent left-looking Doolittle with accurately summed dots.
    l, u = np.eye(n), np.zeros_like(a)
    for k in range(n):
        for j in range(k, n):
            u[k, j] = a[k, j] - math.fsum(l[k, t] * u[t, j] for t in range(k))
        check(u[k, k] != 0, "LU independent pivot breakdown")
        for i in range(k + 1, n):
            l[i, k] = (a[i, k] - math.fsum(l[i, t] * u[t, k] for t in range(k))) / u[
                k, k
            ]
    if np.array_equal(a, np.diag(np.diag(a))):
        np.testing.assert_array_equal(actual, a)
    expected = np.tril(l, -1) + u
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
    observed_l, observed_u = np.tril(actual, -1) + np.eye(n), np.triu(actual)
    check(np.array_equal(np.diag(observed_l), np.ones(n)), "LU unit diagonal")
    residual = float(
        np.linalg.norm(observed_l @ observed_u - a, ord=np.inf)
        / np.linalg.norm(a, ord=np.inf)
    )
    check(residual <= 3e-5, "LU reconstruction residual")
    return {
        "contract": "fixed-packed-LU-and-reconstruction-v1",
        "fixed_accuracy_passed": True,
        "max_factor_abs_error": float(np.max(np.abs(actual - expected))),
        "relative_infinity_residual": residual,
        "residual_limit": 3e-5,
        "L_diagonal": "implicit exact unit diagonal; not a stored device field",
    }


def checkpoint_reference(a, p):
    import numpy as np

    a = np.asarray(a, np.float64).copy()
    n = a.shape[0]
    nt = n // p
    history = np.zeros((p, p, n, 2))
    for k in range(n):
        a[k + 1 :, k] /= a[k, k]
        a[k + 1 :, k + 1 :] -= np.outer(a[k + 1 :, k], a[k, k + 1 :])
        for y in range(p):
            for x in range(p):
                if k < (min(x, y) + 1) * nt:
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
    from mesh_lu import plan, generate

    root = Path(root)
    m, s, r = (
        read(root, name) for name in ("semantic.json", "schedule.json", "results.json")
    )
    batches = read(root, "batches.json")
    check(s == plan(m), "LU schedule regeneration")
    with tempfile.TemporaryDirectory() as temp:
        generate(s, temp)
        for name in ("pe.csl", "layout.csl"):
            check(
                (root / name).read_bytes() == (Path(temp) / name).read_bytes(),
                "LU CSL regeneration",
            )
    check(
        r["success"] and len(r["cases"]) == len(r["diagnostics"]) == len(batches),
        "LU incomplete",
    )
    check(
        r["runtime_instances"] == 1 and r["prepare_barriers"] == len(batches),
        "LU lifecycle",
    )
    check(close(r["cases"], evaluate(m, batches)[0]), "LU HLS/device mismatch")
    p, nt, n = s["P"], s["Nt"], s["N"]
    checks, cycles = [], []
    error, count = 0.0, 0
    for batch, case, diag in zip(batches, r["cases"], r["diagnostics"]):
        check(set(case) == {m["nodes"][2]["host"]}, "LU output ports")
        a = np.asarray(batch[m["nodes"][0]["host"]]).reshape(n, n)
        actual = np.asarray(case[m["nodes"][2]["host"]]).reshape(n, n)
        checks.append(check_factor(a, actual))
        expected = checkpoint_reference(a, p)
        observed = np.asarray(diag["checkpoints"])
        check(observed.shape == expected.shape, "LU checkpoint shape")
        np.testing.assert_allclose(observed, expected, rtol=3e-5, atol=3e-6)
        error = max(error, float(np.max(np.abs(observed - expected))))
        check(
            np.asarray(diag["progress"]).shape == (p, p, 2)
            and np.asarray(diag["timing"]).shape == (p, p, 6),
            "LU diagnostics shape",
        )
        for y in range(p):
            for x in range(p):
                steps = (min(x, y) + 1) * nt
                np.testing.assert_array_equal(
                    observed[y, x, steps:], np.zeros((n - steps, 2))
                )
                check(diag["progress"][y][x] == [steps, 1], f"LU progress p{x}_{y}")
                count += 2 * steps
                words = diag["timing"][y][x]
                ticks = [
                    sum(int(words[i + base]) << (16 * i) for i in range(3))
                    for base in (0, 3)
                ]
                duration = (ticks[1] - ticks[0]) % (1 << 48)
                check(0 < duration < 1 << 32, "LU timestamp bounds")
                cycles.append(duration)
    report = {
        "passed": True,
        "fixed_accuracy_passed": True,
        "actors": p * p,
        "epochs": len(batches),
        "output_values": manifest["expected_output_values"],
        "factor_checks": checks,
        "internal_observations": count,
        "max_checkpoint_abs_error": error,
        "per_pe_factor_cycles": cycles,
        "max_pe_factor_cycles": max(cycles),
        "checkpoint_scope": "Two tile corners after each active pivot; unexecuted entries zero",
        "timing_scope": "Per-PE SDK simulator factorization including communication, excluding prepare/host I/O; no hardware claim",
        "source_sha256": manifest["source_sha256"],
    }
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
