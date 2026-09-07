"""Execute generated SUMMA CSL and audit every resident partial tile."""

import json
import os
from pathlib import Path
import subprocess
from mesh_common import pack_tiles, unpack_tiles


def read(root, name):
    return json.loads((root / name).read_text())


def run(root):
    import numpy as np
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime,
        MemcpyOrder,
        MemcpyDataType,
    )

    root = Path(root).resolve()
    s, m = read(root, "schedule.json"), read(root, "semantic.json")
    cannon = s["profile"] == "mesh_cannon.v1"
    order = "C" if cannon else "F"
    p, mt, kt, nt = (s[k] for k in ("P", "Mt", "Kt", "Nt"))
    os.chdir(root)
    cmd = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        f"--fabric-dims={p+7},{p+2}",
        "--fabric-offsets=4,1",
        f"--params=P:{p},Mt:{mt},Kt:{kt},Nt:{nt}",
        "-o=out",
        "--memcpy",
        "--channels=1",
        "--width-west-buf=0",
        "--width-east-buf=0",
    ]
    (root / "sdk-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    subprocess.run(cmd, check=True)
    from mesh_common import sdk_runtime

    runner = sdk_runtime(root)
    symbols = ["A", "B", "C", "history", "timing"]
    if cannon:
        symbols += ["witness", "progress", "queue_last", "total_timing"]
    ids = {name: runner.get_id(name) for name in symbols}
    runner.load()
    runner.run()
    r = {
        "success": False,
        "cases": [],
        "diagnostics": [],
        "runtime_instances": 1,
        "launches": [],
    }

    def save():
        (root / "results.json").write_text(json.dumps(r) + "\n")

    def get(name, length, short=False):
        buf = np.zeros(p * p * length, np.uint32 if short else np.float32)
        runner.memcpy_d2h(
            buf,
            ids[name],
            0,
            0,
            p,
            p,
            length,
            streaming=False,
            data_type=(
                MemcpyDataType.MEMCPY_16BIT if short else MemcpyDataType.MEMCPY_32BIT
            ),
            order=MemcpyOrder.COL_MAJOR,
            nonblock=False,
        )
        return buf.reshape(p, p, length, order="F")

    try:
        for epoch, batch in enumerate(read(root, "batches.json")):
            for name, node in zip(("A", "B"), m["nodes"][:2]):
                matrix = np.asarray(batch[node["host"]], np.float32).reshape(
                    node["shape"]
                )
                if cannon:
                    from mesh_cannon import pack

                    tiles = pack(matrix, p, name)
                else:
                    tiles = pack_tiles(matrix, p, p, order)
                runner.memcpy_h2d(
                    ids[name],
                    tiles.ravel(order="F"),
                    0,
                    0,
                    p,
                    p,
                    tiles.shape[-1],
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_32BIT,
                    order=MemcpyOrder.COL_MAJOR,
                    nonblock=False,
                )
            (root / "runtime-stage.json").write_text(
                json.dumps(dict(epoch=epoch, operation="launch main"))
            )
            runner.launch("main", nonblock=False)
            r["launches"].append("main")
            final = unpack_tiles(get("C", mt * nt), mt, nt, order)
            r["cases"].append({m["nodes"][3]["host"]: final.ravel().tolist()})
            r["diagnostics"].append(
                {
                    "history": get("history", p * mt * nt)
                    .reshape(p, p, p, mt * nt)
                    .tolist(),
                    "timing": get("timing", p * 6, True).reshape(p, p, p, 6).tolist(),
                }
            )
            if cannon:
                r["diagnostics"][-1].update(
                    witness=get("witness", p * 4).reshape(p, p, p, 4).tolist(),
                    progress=get("progress", 4, True).tolist(),
                    queue_last=get("queue_last", 2, True).tolist(),
                    total_timing=get("total_timing", 6, True).tolist(),
                )
            save()
            print(s["profile"], "EPOCH", epoch + 1, "COMPLETE", flush=True)
    finally:
        runner.stop()
    r["success"] = True
    save()


def audit(root, manifest):
    import numpy as np
    import tempfile
    from frontend import check
    from roundoff import check_matrix_roundoff
    from ir import evaluate
    from mesh_gemm import plan, generate

    root = Path(root)
    m, s, r = (
        read(root, "semantic.json"),
        read(root, "schedule.json"),
        read(root, "results.json"),
    )
    batches = read(root, "batches.json")
    cannon = s["profile"] == "mesh_cannon.v1"
    if cannon:
        from mesh_cannon import plan, generate
    order = "C" if cannon else "F"
    check(s == plan(m), "matrix schedule regeneration")
    with tempfile.TemporaryDirectory() as temp:
        generate(s, temp)
        for name in ("pe.csl", "layout.csl"):
            check(
                (root / name).read_bytes() == (Path(temp) / name).read_bytes(),
                "SUMMA CSL regeneration",
            )
    check(r["success"] and len(r["diagnostics"]) == len(batches), "SUMMA incomplete")
    expected_cases = evaluate(m, batches)[0]
    check(len(r["cases"]) == len(expected_cases), "SUMMA epoch count")
    final_roundoff = []
    for batch, actual, expected in zip(batches, r["cases"], expected_cases):
        a, b = [
            np.asarray(batch[n["host"]], np.float64).reshape(n["shape"])
            for n in m["nodes"][:2]
        ]
        outname = m["nodes"][3]["host"]
        check(set(actual) == {outname}, "SUMMA output ports")
        observed = np.asarray(actual[outname]).reshape(a.shape[0], b.shape[1])
        same_order = np.asarray(expected[outname]).reshape(a.shape[0], b.shape[1])
        reference = a @ b
        final_roundoff.append(
            {
                "vs_hls_f32": check_matrix_roundoff(
                    a, b, observed, same_order, reference_is_f32=True
                ),
                "vs_float64": check_matrix_roundoff(a, b, observed, reference),
                "hls_vs_float64": check_matrix_roundoff(a, b, same_order, reference),
            }
        )
    p, mt, kt, nt = (s[k] for k in ("P", "Mt", "Kt", "Nt"))
    count = 0
    error = 0.0
    normalized_error = 0.0
    partial_fixed_accuracy = True
    durations = []
    total_durations = []
    for epoch, (batch, diag, final_case) in enumerate(
        zip(batches, r["diagnostics"], r["cases"])
    ):
        a, b = [
            np.asarray(batch[n["host"]], np.float64).reshape(n["shape"])
            for n in m["nodes"][:2]
        ]
        if cannon:
            check(
                r["runtime_instances"] == 1
                and r["launches"] == ["main"] * len(batches),
                "Cannon lifecycle",
            )
            np.testing.assert_array_equal(
                np.asarray(diag["progress"]),
                np.tile([p, p - 1, p - 1, epoch + 1], (p, p, 1)),
            )
            check(
                np.asarray(diag["queue_last"]).shape == (p, p, 2),
                "Cannon queue mask shape",
            )
            for mask in np.asarray(diag["queue_last"]).ravel():
                check(int(mask) & 12 == 12, "Cannon queues drained")
            t = np.asarray(diag["total_timing"], dtype=np.int64)
            check(
                t.shape == (p, p, 6) and np.all((t >= 0) & (t < 65536)),
                "Cannon total timestamp words",
            )
            ticks = sum(
                (t[:, :, i + 3] - t[:, :, i]) * (1 << (16 * i)) for i in range(3)
            )
            check(
                np.all((ticks > 0) & (ticks < 2**32)), "Cannon total timestamp interval"
            )
            total_durations.append(ticks.tolist())
        for row in range(p):
            for col in range(p):
                for step in range(p):
                    end = (step + 1) * kt
                    indices = (
                        [
                            q
                            for rstep in range(step + 1)
                            for q in range(
                                ((row + col + rstep) % p) * kt,
                                ((row + col + rstep) % p + 1) * kt,
                            )
                        ]
                        if cannon
                        else list(range(end))
                    )
                    left = a[row * mt : (row + 1) * mt, indices]
                    right = b[indices, col * nt : (col + 1) * nt]
                    expected = left @ right
                    if cannon:
                        kblock = (row + col + step) % p
                        np.testing.assert_array_equal(
                            diag["witness"][row][col][step],
                            [
                                a[row * mt, kblock * kt],
                                a[(row + 1) * mt - 1, (kblock + 1) * kt - 1],
                                b[kblock * kt, col * nt],
                                b[(kblock + 1) * kt - 1, (col + 1) * nt - 1],
                            ],
                        )
                    actual = np.asarray(diag["history"][row][col][step]).reshape(
                        mt, nt, order=order
                    )
                    if step == p - 1:
                        final = np.asarray(final_case[m["nodes"][3]["host"]]).reshape(
                            s["matrix_rows"], s["matrix_cols"]
                        )
                        np.testing.assert_array_equal(
                            actual,
                            final[row * mt : (row + 1) * mt, col * nt : (col + 1) * nt],
                        )
                    rounding = check_matrix_roundoff(
                        left,
                        right,
                        actual,
                        expected,
                    )
                    partial_fixed_accuracy = (
                        partial_fixed_accuracy
                        and rounding["old_fixed_tolerance_passed"]
                    )
                    normalized_error = max(
                        normalized_error, rounding["max_error_over_bound"]
                    )
                    error = max(error, float(np.max(np.abs(actual - expected))))
                    words = diag["timing"][row][col][step]
                    start = sum(int(words[i]) << (16 * i) for i in range(3))
                    stop = sum(int(words[i + 3]) << (16 * i) for i in range(3))
                    cycles = (stop - start) % (1 << 48)
                    check(0 < cycles < 1 << 32, "SUMMA timestamp bounds")
                    durations.append(cycles)
                    count += mt * nt
    report = {
        "passed": True,
        "arithmetic_roundoff_passed": True,
        "fixed_accuracy_passed": partial_fixed_accuracy
        and all(v["vs_float64"]["old_fixed_tolerance_passed"] for v in final_roundoff),
        "acceptance_contract": "componentwise-f32-dot-v1 with exact final/history agreement; fixed 3e-5/3e-6 accuracy is reported separately",
        "actors": p * p,
        "epochs": len(batches),
        "rounds": p,
        "output_values": manifest["expected_output_values"],
        "internal_observations": count,
        "max_partial_abs_error": error,
        "max_partial_error_over_bound": normalized_error,
        "final_roundoff": final_roundoff,
        "compute_cycles_per_pe_round": durations,
        "total_cycles_per_epoch_pe": total_durations,
        "schedule": "Cannon cyclic blocks" if cannon else "SUMMA broadcasts",
        "compute_cycles_median": float(np.median(durations)),
        "timing_scope": "Per-PE per-round local matrix multiply only, excluding broadcast and history copy; SDK simulator, not hardware.",
        "source_sha256": manifest["source_sha256"],
    }
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
