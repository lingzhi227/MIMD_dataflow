"""SDK execution and independent local-partial audit for mesh GEMV."""

import json
import os
from pathlib import Path
import subprocess
from mesh_common import pack_tiles


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
    nr, nc, mt, nt = (s[k] for k in ("kernel_rows", "kernel_cols", "Mt", "Nt"))
    rows, cols = s["matrix_rows"], s["matrix_cols"]
    os.chdir(root)
    command = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        f"--fabric-dims={nc+7},{nr+2}",
        "--fabric-offsets=4,1",
        f"--params=kernel_rows:{nr},kernel_cols:{nc},matrix_rows:{rows},matrix_cols:{cols}",
        "-o=out",
        "--memcpy",
        "--channels=1",
        "--width-west-buf=0",
        "--width-east-buf=0",
    ]
    (root / "sdk-command.json").write_text(json.dumps(command, indent=2) + "\n")
    subprocess.run(command, check=True)
    from mesh_common import sdk_runtime

    runner = sdk_runtime(root)
    ids = {
        n: runner.get_id(n)
        for n in ("A", "x", "y", "partial", "x_tile", "compute_time")
    }
    runner.load()
    runner.run()
    report = {"success": False, "cases": [], "diagnostics": []}

    def save():
        (root / "results.json").write_text(json.dumps(report) + "\n")

    def receive(name, x, y, w, h, length, short=False):
        buf = np.zeros(w * h * length, np.uint32 if short else np.float32)
        runner.memcpy_d2h(
            buf,
            ids[name],
            x,
            y,
            w,
            h,
            length,
            streaming=False,
            data_type=(
                MemcpyDataType.MEMCPY_16BIT if short else MemcpyDataType.MEMCPY_32BIT
            ),
            order=MemcpyOrder.COL_MAJOR,
            nonblock=False,
        )
        return buf.reshape(h, w, length, order="F")

    try:
        for epoch, b in enumerate(read(root, "batches.json")):
            matrix = np.asarray(b[m["nodes"][0]["host"]], np.float32).reshape(
                rows, cols
            )
            tiled = pack_tiles(matrix, nr, nc, "C")
            runner.memcpy_h2d(
                ids["A"],
                tiled.ravel(order="F"),
                0,
                0,
                nc,
                nr,
                mt * nt,
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.COL_MAJOR,
                nonblock=False,
            )
            runner.memcpy_h2d(
                ids["x"],
                np.asarray(b[m["nodes"][1]["host"]], np.float32),
                0,
                0,
                1,
                1,
                cols,
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.COL_MAJOR,
                nonblock=False,
            )
            runner.launch("main", nonblock=False)
            report["cases"].append(
                {
                    m["nodes"][3]["host"]: receive("y", nc - 1, nr - 1, 1, 1, rows)
                    .ravel()
                    .tolist()
                }
            )
            report["diagnostics"].append(
                {
                    "partial": receive("partial", 0, 0, nc, nr, mt).tolist(),
                    "x_tile": receive("x_tile", 0, 0, nc, nr, nt).tolist(),
                    "compute_time": receive(
                        "compute_time", 0, 0, nc, nr, 6, True
                    ).tolist(),
                }
            )
            save()
            print("MESH GEMV EPOCH", epoch + 1, "COMPLETE", flush=True)
    finally:
        runner.stop()
    report["success"] = True
    save()


def audit(root, manifest):
    import numpy as np
    from frontend import check
    from ir import evaluate
    from float32 import close
    from mesh_gemv import plan, generate
    import tempfile

    root = Path(root)
    m, s, r = (
        read(root, "semantic.json"),
        read(root, "schedule.json"),
        read(root, "results.json"),
    )
    batches = read(root, "batches.json")
    check(s == plan(m), "mesh schedule regeneration")
    with tempfile.TemporaryDirectory() as tmp:
        generate(s, tmp)
        for name in ("pe.csl", "layout.csl"):
            check(
                (root / name).read_bytes() == (Path(tmp) / name).read_bytes(),
                "mesh CSL regeneration",
            )
    check(r["success"] and len(r["diagnostics"]) == len(batches), "mesh incomplete")
    check(close(r["cases"], evaluate(m, batches)[0]), "mesh output vs HLS IR")
    nr, nc, mt, nt = (s[k] for k in ("kernel_rows", "kernel_cols", "Mt", "Nt"))
    observations = 0
    durations = []
    error = 0.0
    for batch, diag in zip(batches, r["diagnostics"]):
        a = np.asarray(batch[m["nodes"][0]["host"]], np.float64).reshape(
            s["matrix_rows"], s["matrix_cols"]
        )
        x = np.asarray(batch[m["nodes"][1]["host"]], np.float64)
        for y in range(nr):
            for col in range(nc):
                xp = x[col * nt : (col + 1) * nt]
                np.testing.assert_array_equal(diag["x_tile"][y][col], xp)
                expected = a[y * mt : (y + 1) * mt, col * nt : (col + 1) * nt] @ xp
                actual = np.asarray(diag["partial"][y][col])
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
                error = max(error, float(np.max(np.abs(actual - expected))))
                words = diag["compute_time"][y][col]
                start = sum(int(words[i]) << (16 * i) for i in range(3))
                end = sum(int(words[i + 3]) << (16 * i) for i in range(3))
                cycles = (end - start) % (1 << 48)
                check(0 < cycles < 1 << 32, "mesh compute timing bounds")
                durations.append(cycles)
                observations += mt + nt
    result = {
        "passed": True,
        "actors": nr * nc,
        "epochs": len(batches),
        "output_values": manifest["expected_output_values"],
        "internal_observations": observations,
        "max_partial_abs_error": error,
        "compute_cycles_per_pe": durations,
        "compute_cycles_median": float(np.median(durations)),
        "timing_scope": "Per-PE local matvec only; excludes scatter/broadcast/reduce/gather and host I/O. Simulator, not hardware.",
        "source_sha256": manifest["source_sha256"],
    }
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
