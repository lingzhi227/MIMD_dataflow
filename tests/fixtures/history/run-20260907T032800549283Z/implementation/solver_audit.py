"""Shared provenance, original sparse operator and resident protocol audit services."""

import json, math, tempfile
from pathlib import Path
import numpy as np
from frontend import check
from mesh_spmv_sdk import read


def load(root):
    from mesh_cg import generate, TEMPLATES
    from planner import plan
    from mesh_cg import packing as cg_packing
    from mesh_power import packing as power_packing
    from resident_abi import schema

    root = Path(root)
    m = read(root, "semantic.json")
    s = read(root, "schedule.json")
    r = read(root, "results.json")
    b = read(root, "batches.json")
    packing = power_packing if s.get("solver") == "power" else cg_packing
    check(read(root, "host-abi.json") == schema(s), "resident host ABI regeneration")
    check(s == plan(m), "resident schedule regeneration")
    with tempfile.TemporaryDirectory() as tmp:
        generate(s, tmp)
        for name in TEMPLATES:
            check(
                (root / name).read_bytes() == (Path(tmp) / name).read_bytes(),
                "resident CSL regeneration " + name,
            )
    check(read(root, "sparse-packing.json") == packing(m, b), "resident input packing")
    check(
        r["success"]
        and r["runtime_instances"] == 1
        and len(r["cases"]) == len(r["diagnostics"]) == len(b) == m["epochs"]
        and r["launches"] == ["f_cg"] * len(b),
        "resident lifecycle",
    )
    return m, s, r, b


def apply(a, x):
    terms = [[] for _ in x]
    for row, col, value in a.entries():
        terms[row].append(float(value) * float(x[col]))
    return np.array([math.fsum(v) for v in terms])


def record(m, s, raw, d):
    o = {n["inputs"][0].split(".")[1]: raw[n["host"]] for n in m["nodes"][8:]}
    n = s["N"]
    cap = s["max_iterations"]
    h, w = s["rows"], s["cols"]
    for field, diag, length in [
        ("reason", "cg_reason", 1),
        ("iterations", "cg_iterations", 1),
        ("residual_squared", "cg_history", cap + 1),
        ("true_residual_norm", "cg_true_norm", 1),
    ]:
        data = np.asarray(d[diag])
        check(data.shape == (h, w, length), "resident record shape " + diag)
        np.testing.assert_array_equal(data, np.broadcast_to(o[field], data.shape))
    x = np.asarray(o["solution"])
    check(x.shape == (n,) and np.all(np.isfinite(x)), "resident solution domain")
    check(
        np.asarray(d["cg_solution"]).shape == (h, w, s["geometry"]["local_vec_sz"]),
        "resident solution ownership",
    )
    np.testing.assert_array_equal(x, np.asarray(d["cg_solution"]).ravel())
    check(
        all(type(o[f][0]) is int for f in ["reason", "iterations"]),
        "resident integer result",
    )
    return o


def replicated(d, key, s, length):
    a = np.asarray(d[key])
    check(a.shape == (s["rows"], s["cols"], length), "resident replica shape " + key)
    np.testing.assert_array_equal(a, np.broadcast_to(a[0, 0], a.shape))
    check(np.all(np.isfinite(a)), "resident finite diagnostic " + key)
    return a[0, 0]


def protocol(s, a, x, d, epoch, spmv_calls, scalar_calls, k):
    # No operator was launched for a zero-step power call; buffers remain stale.
    active_operator = spmv_calls > 0
    if active_operator:
        np.testing.assert_allclose(
            np.asarray(d["final_ax"]).ravel(), apply(a, x), rtol=3e-5, atol=3e-6
        )
    progress = np.asarray(d["progress"])
    check(progress.shape == (s["rows"], s["cols"], 11), "resident train progress shape")
    np.testing.assert_array_equal(progress[:, :, :10], 0)
    np.testing.assert_array_equal(progress[:, :, 10], epoch + 1)
    cp = replicated(d, "cg_progress", s, 6)
    np.testing.assert_array_equal(
        cp, [spmv_calls, spmv_calls, scalar_calls, k, epoch + 1, 1 + 3 * spmv_calls]
    )
    for mask in np.asarray(d["cg_queue_last"]).ravel():
        check(int(mask) & 252 == 252, "resident drained shared queues")
    g = s["geometry"]
    terms = [[{} for _ in range(s["cols"])] for _ in range(s["rows"])]
    for row, col, value in a.entries():
        terms[row // g["block_rows"]][col // g["block_cols"]].setdefault(
            row, []
        ).append(float(value) * float(x[col]))
    partial = np.asarray(d["partial"])
    check(partial.shape == (s["rows"], s["cols"], 2), "resident partial witness shape")
    for y in range(s["rows"]):
        for xpe in range(s["cols"]):
            ids = sorted(terms[y][xpe])
            expected = (
                [math.fsum(terms[y][xpe][row]) for row in (ids[0], ids[-1])]
                if ids
                else [0.0, 0.0]
            )
            if active_operator:
                np.testing.assert_allclose(
                    partial[y, xpe], expected, rtol=3e-5, atol=3e-6
                )
    t = np.asarray(d["cg_timing"])
    check(
        t.shape == (s["rows"], s["cols"], 6)
        and np.issubdtype(t.dtype, np.integer)
        and np.all((t >= 0) & (t < 65536)),
        "resident timestamp words",
    )
    ticks = sum((t[:, :, i + 3] - t[:, :, i]) * (1 << (16 * i)) for i in range(3))
    check(np.all((ticks > 0) & (ticks < 2**32)), "resident timestamp interval")
    return int(ticks.max())
