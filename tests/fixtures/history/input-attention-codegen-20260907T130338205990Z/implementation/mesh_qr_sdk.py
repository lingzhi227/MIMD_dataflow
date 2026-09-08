"""R-only QR audit: reference/Gram/derived-Q plus sampled device rotations."""

import json
from pathlib import Path


def read(root, name):
    return json.loads((Path(root) / name).read_text())


def run(root):
    from factor_sdk import run_factor

    return run_factor(
        root,
        "A",
        "start",
        {"M": "M", "N": "N", "grid_height": "rows", "grid_width": "cols"},
        [
            ("timing", 6, True, [6]),
            ("progress", 2, True, [2]),
            ("witnesses", 128, False, [16, 8]),
        ],
    )


def check_factor(a, actual):
    import numpy as np
    from frontend import check

    a, actual = np.asarray(a, np.float64), np.asarray(actual, np.float64)
    check(
        a.ndim == 2 and a.shape[0] >= a.shape[1] and actual.shape == a.shape, "QR shape"
    )
    check(np.all(np.isfinite(a)) and np.all(np.isfinite(actual)), "QR finite")
    m, n = a.shape
    condition = float(np.linalg.cond(a))
    check(condition <= 10000, "QR full-rank condition limit 10000")
    _, reference = np.linalg.qr(a, mode="reduced")
    top = actual[:n]
    check(
        np.all(np.abs(np.diag(top)) > 2**-17 * np.max(np.abs(a))), "QR diagonal margin"
    )
    signs = np.sign(np.diag(top)) * np.sign(np.diag(reference))
    aligned = np.zeros_like(a)
    aligned[:n] = reference * signs[:, None]
    np.testing.assert_allclose(actual, aligned, rtol=3e-5, atol=3e-6)
    gram = float(
        np.linalg.norm(actual.T @ actual - a.T @ a, ord=np.inf)
        / np.linalg.norm(a.T @ a, ord=np.inf)
    )
    check(gram <= 3e-5, "QR Gram residual")
    # Q is reconstructed through a solve only within the checked conditioning domain.
    # It is not an output observed on the device.
    q = np.linalg.solve(top.T, a.T).T
    orthogonality = float(np.linalg.norm(q.T @ q - np.eye(n), ord=np.inf))
    check(orthogonality <= 3e-5, "QR derived-Q orthogonality")
    reconstruction = float(
        np.linalg.norm(q @ top - a, ord=np.inf) / np.linalg.norm(a, ord=np.inf)
    )
    return {
        "contract": "fixed-R-sign-equivalence-Gram-derived-Q-v1",
        "fixed_accuracy_passed": True,
        "condition2": condition,
        "row_signs_to_reference": signs.tolist(),
        "max_factor_abs_error": float(np.max(np.abs(actual - aligned))),
        "relative_gram_residual": gram,
        "derived_Q_orthogonality": orthogonality,
        "derived_Q_reconstruction": reconstruction,
        "Q_scope": "Host-derived via solve; not emitted device Q. Gram and R checks are independent requirements.",
    }


def audit_rotations(witnesses, total, expected_kinds=None):
    import numpy as np
    from frontend import check

    w = np.asarray(witnesses, np.float64)
    check(
        w.shape == (16, 8)
        and np.all(np.isfinite(w))
        and type(total) is int
        and 0 < total < 65536,
        "QR witness shape/count",
    )
    from qr_schedule import sampled_slots

    slots = sampled_slots(total)
    check(
        expected_kinds is None or total == len(expected_kinds),
        "QR symbolic rotation count",
    )
    max_ratio = 0.0
    u = 2**-24
    gamma = 3 * u / (1 - 3 * u)
    for slot, row in enumerate(w):
        if slot not in slots:
            np.testing.assert_array_equal(row, np.zeros(8))
            continue
        kind, serial, c, s, a, b, top, bottom = row
        check(kind in (1, 2, 3) and serial == slots[slot], "QR sampled serial/kind")
        if expected_kinds is not None:
            check(kind == expected_kinds[int(serial)], "QR symbolic rotation role")
        check(abs(c * c + s * s - 1) <= 4e-6, "QR Givens unit norm")
        for wanted, observed, x, y in (
            (kind in (1, 2), top, c * a, -s * b),
            (kind in (1, 3), bottom, s * a, c * b),
        ):
            if not wanted:
                check(observed == 0, "QR nonlocal witness field")
                continue
            budget = gamma * (abs(x) + abs(y)) + 3 * np.finfo(np.float32).tiny
            error = abs(observed - (x + y))
            check(error <= budget, "QR rotation arithmetic envelope")
            max_ratio = max(max_ratio, error / budget)
    return {"samples": len(slots), "max_rotation_error_over_bound": max_ratio}


def audit(root, manifest):
    import tempfile
    import numpy as np
    from frontend import check
    from mesh_qr import plan, generate
    from qr_schedule import rotations

    root = Path(root)
    m, s, r = (
        read(root, name) for name in ("semantic.json", "schedule.json", "results.json")
    )
    batches = read(root, "batches.json")
    check(s == plan(m), "QR schedule regeneration")
    with tempfile.TemporaryDirectory() as temp:
        generate(s, temp)
        for name in ("pe.csl", "layout.csl"):
            check(
                (root / name).read_bytes() == (Path(temp) / name).read_bytes(),
                "QR CSL regeneration",
            )
    check(
        r["success"] and len(r["cases"]) == len(r["diagnostics"]) == len(batches),
        "QR incomplete",
    )
    check(
        r["runtime_instances"] == 1 and r["prepare_barriers"] == len(batches),
        "QR lifecycle",
    )
    checks, cycles, rotation_checks = [], [], []
    counts = None
    for batch, case, diag in zip(batches, r["cases"], r["diagnostics"]):
        check(set(case) == {m["nodes"][2]["host"]}, "QR output ports")
        checks.append(
            check_factor(
                np.asarray(batch[m["nodes"][0]["host"]]).reshape(s["M"], s["N"]),
                np.asarray(case[m["nodes"][2]["host"]]).reshape(s["M"], s["N"]),
            )
        )
        rows, cols = s["rows"], s["cols"]
        check(
            np.asarray(diag["progress"]).shape == (rows, cols, 2)
            and np.asarray(diag["timing"]).shape == (rows, cols, 6)
            and np.asarray(diag["witnesses"]).shape == (rows, cols, 16, 8),
            "QR diagnostics shapes",
        )
        current = []
        for y in range(rows):
            for x in range(cols):
                total, done = diag["progress"][y][x]
                check(done == 1, "QR PE completion")
                current.append(total)
                expected_kinds = rotations(rows, cols, s["Nt"], x, y)
                check(total == len(expected_kinds), "QR symbolic rotation count")
                if s.get("instrumentation", "sampled") == "counters":
                    np.testing.assert_array_equal(
                        diag["witnesses"][y][x], np.zeros((16, 8))
                    )
                else:
                    rotation_checks.append(
                        audit_rotations(
                            diag["witnesses"][y][x],
                            total,
                            rotations(rows, cols, s["Nt"], x, y),
                        )
                    )
                words = diag["timing"][y][x]
                start, end = [
                    sum(int(words[i + base]) << (16 * i) for i in range(3))
                    for base in (0, 3)
                ]
                duration = (end - start) % (1 << 48)
                check(0 < duration < 1 << 32, "QR timestamp bounds")
                cycles.append(duration)
        check(
            counts is None or current == counts,
            "QR geometry-dependent counts changed between calls",
        )
        counts = current
    report = {
        "passed": True,
        "fixed_accuracy_passed": True,
        "actors": s["rows"] * s["cols"],
        "epochs": len(batches),
        "output_values": manifest["expected_output_values"],
        "factor_checks": checks,
        "rotation_checks": rotation_checks,
        "per_pe_rotation_counts": counts,
        "instrumentation": s.get("instrumentation", "sampled"),
        "rotation_witnesses": s.get("instrumentation", "sampled") == "sampled",
        "sampled_rotations": sum(v["samples"] for v in rotation_checks),
        "per_pe_factor_cycles": cycles,
        "max_pe_factor_cycles": max(cycles),
        "timing_scope": "Per-PE SDK simulator factorization including communication; excludes prepare and host I/O",
        "source_sha256": manifest["source_sha256"],
    }
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
