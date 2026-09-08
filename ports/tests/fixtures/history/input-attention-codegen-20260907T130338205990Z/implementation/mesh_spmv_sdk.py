"""SDK train SpMV host binding and independent original-entry numerical audit."""

import json
import math
from pathlib import Path


def read(root, name):
    return json.loads((Path(root) / name).read_text())


def check_result(rows, cols, offsets, indices, values, x, actual):
    import numpy as np
    from frontend import check

    # Direct original entry accumulation, independent of partition/packed rows.
    terms = [[] for _ in range(rows)]
    for c in range(cols):
        for p in range(offsets[c], offsets[c + 1]):
            terms[indices[p]].append(float(values[p]) * float(x[c]))
    reference = np.asarray([math.fsum(t) for t in terms])
    got = np.asarray(actual)
    check(
        got.shape == (rows,) and np.all(np.isfinite(got)), "sparse output shape/finite"
    )
    np.testing.assert_allclose(got, reference, rtol=3e-5, atol=3e-6)
    if not any(values) or not any(x):
        np.testing.assert_array_equal(got, np.zeros(rows))
    return dict(
        contract="fixed-original-entry-SpMV-v1",
        fixed_accuracy_passed=True,
        max_abs_error=float(np.max(np.abs(got - reference))),
    )


def run(root, diagnostics=True):
    import os
    import subprocess
    import numpy as np
    from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder
    from mesh_common import sdk_runtime
    from sparse_storage import distribute_x, gather_y

    root = Path(root).resolve()
    s, m = read(root, "schedule.json"), read(root, "semantic.json")
    rows, cols = s["rows"], s["cols"]
    params = dict(
        prows=rows,
        pcols=cols,
        nrows=s["M"],
        ncols=s["N"],
        max_local_nnz=s["capacity"]["nnz"],
        max_local_nnz_cols=s["capacity"]["columns"],
        max_local_nnz_rows=s["capacity"]["rows"],
    )
    params.update(
        {
            k: s["geometry"][k]
            for k in ("local_vec_sz", "local_out_vec_sz", "y_pad_start_row_idx")
        }
    )
    os.chdir(root)
    cmd = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        f"--fabric-dims={cols+7},{rows+2}",
        "--fabric-offsets=4,1",
        "--params=" + ",".join(f"{k}:{v}" for k, v in params.items()),
        "-o=out",
        "--memcpy",
        "--channels=1",
        "--width-west-buf=0",
        "--width-east-buf=0",
    ]
    (root / "sdk-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    subprocess.run(cmd, check=True)

    def mark(label):
        (root / "runtime-stage.json").write_text(
            json.dumps({"operation": label}) + "\n"
        )
        print("SPARSE SDK", label, flush=True)

    mark("load runtime")
    runner = sdk_runtime(root)
    ids = {
        name: runner.get_id(name)
        for name in list(s["extents"])
        + [
            "x_tx_buf",
            "y_local_buf",
            "time_buf_u16",
        ]
    }
    if diagnostics:
        ids.update(
            {name: runner.get_id(name) for name in ("hls_progress", "hls_partial")}
        )
    runner.load()
    runner.run()
    results = dict(
        success=False,
        cases=[],
        diagnostics=[],
        runtime_instances=1,
    )

    def save():
        (root / "results.json").write_text(json.dumps(results) + "\n")

    def put(name, data, short=False):
        mark("H2D " + name)
        data = np.asarray(data, np.uint32 if short else np.float32)
        runner.memcpy_h2d(
            ids[name],
            data.ravel(order="F"),
            0,
            0,
            cols,
            rows,
            data.shape[2],
            streaming=False,
            data_type=(
                MemcpyDataType.MEMCPY_16BIT if short else MemcpyDataType.MEMCPY_32BIT
            ),
            order=MemcpyOrder.COL_MAJOR,
            nonblock=False,
        )

    def get(name, length, short=False):
        mark("D2H " + name)
        data = np.zeros(rows * cols * length, np.uint32 if short else np.float32)
        runner.memcpy_d2h(
            data,
            ids[name],
            0,
            0,
            cols,
            rows,
            length,
            streaming=False,
            data_type=(
                MemcpyDataType.MEMCPY_16BIT if short else MemcpyDataType.MEMCPY_32BIT
            ),
            order=MemcpyOrder.COL_MAJOR,
            nonblock=False,
        )
        return data.reshape(rows, cols, length, order="F").tolist()

    def launch(name, *args, **kwargs):
        mark("launch " + name)
        runner.launch(name, *args, **kwargs)

    try:
        launch("f_enable_tsc", nonblock=False)
        packs = read(root, "sparse-packing.json")
        for epoch, batch in enumerate(read(root, "batches.json")):
            for name in s["extents"]:
                put(
                    name,
                    [[tile[name] for tile in row] for row in packs[epoch]["tiles"]],
                    name != "mat_vals_buf",
                )
            put(
                "x_tx_buf",
                distribute_x(batch[m["nodes"][3]["host"]], s["M"], s["N"], rows, cols),
            )
            launch("f_tic", nonblock=True)
            launch("f_spmv", nonblock=False)
            launch("f_toc", nonblock=False)
            launch("f_memcpy_timestamps", nonblock=False)
            output = get("y_local_buf", s["geometry"]["local_out_vec_sz"])
            if diagnostics:
                launch("hls_capture", nonblock=False)
            results["cases"].append(
                {m["nodes"][-1]["host"]: gather_y(output, s["M"], s["N"], rows, cols)}
            )
            results["diagnostics"].append(
                dict(
                    output_tiles=output,
                    progress=get("hls_progress", 11, True) if diagnostics else None,
                    partial=get("hls_partial", 2) if diagnostics else None,
                    timing=get("time_buf_u16", 6, True),
                )
            )
            save()
            print("mesh_spmv.v1 EPOCH", epoch + 1, "COMPLETE", flush=True)
    finally:
        runner.stop()
    results["success"] = True
    save()


def audit(root, manifest):
    import tempfile
    import numpy as np
    from frontend import check
    from mesh_spmv import plan, generate, matrix, TEMPLATES
    from sparse_storage import partition, Capacity, gather_y

    root = Path(root)
    m, s, r = (
        read(root, name) for name in ("semantic.json", "schedule.json", "results.json")
    )
    check(s == plan(m), "sparse schedule regeneration")
    with tempfile.TemporaryDirectory() as tmp:
        generate(s, tmp)
        for name in TEMPLATES:
            check(
                (root / name).read_bytes() == (Path(tmp) / name).read_bytes(),
                "sparse CSL regeneration " + name,
            )
    batches = read(root, "batches.json")
    packs = read(root, "sparse-packing.json")
    check(
        r["success"]
        and r["runtime_instances"] == 1
        and len(r["cases"]) == len(r["diagnostics"]) == len(batches) == m["epochs"],
        "sparse lifecycle",
    )
    checks = []
    timings = []
    for epoch, (batch, out, d) in enumerate(zip(batches, r["cases"], r["diagnostics"])):
        a = matrix(m, batch)
        x = batch[m["nodes"][3]["host"]]
        check(
            packs[epoch]
            == partition(a, s["rows"], s["cols"], Capacity(**s["capacity"])),
            "sparse packing regeneration",
        )
        actual = out[m["nodes"][-1]["host"]]
        check(
            actual == gather_y(d["output_tiles"], a.rows, a.cols, s["rows"], s["cols"]),
            "sparse output unpack",
        )
        checks.append(
            check_result(
                a.rows, a.cols, a.column_offsets, a.row_indices, a.values, x, actual
            )
        )
        progress = np.asarray(d["progress"])
        check(progress.shape == (s["rows"], s["cols"], 11), "sparse progress shape")
        np.testing.assert_array_equal(progress[:, :, :10], 0)
        np.testing.assert_array_equal(progress[:, :, 10], epoch + 1)
        partial = np.asarray(d["partial"])
        check(partial.shape == (s["rows"], s["cols"], 2), "sparse witness shape")
        # Select witness rows directly from original entries, not packed buffers.
        terms = [[{} for _ in range(s["cols"])] for _ in range(s["rows"])]
        g = s["geometry"]
        for row, col, val in a.entries():
            terms[row // g["block_rows"]][col // g["block_cols"]].setdefault(
                row, []
            ).append(float(val) * float(x[col]))
        intervals = []
        for py in range(s["rows"]):
            for px in range(s["cols"]):
                rows = sorted(terms[py][px])
                expected = (
                    [math.fsum(terms[py][px][row]) for row in (rows[0], rows[-1])]
                    if rows
                    else [0.0, 0.0]
                )
                np.testing.assert_allclose(
                    partial[py, px], expected, rtol=3e-5, atol=3e-6
                )
                for z, val in enumerate(d["output_tiles"][py][px]):
                    local = px * g["local_out_vec_sz"] + z
                    if (
                        local >= g["block_rows"]
                        or py * g["block_rows"] + local >= a.rows
                    ):
                        check(val == 0, "sparse output padding exact zero")
                words = d["timing"][py][px]
                check(
                    len(words) == 6
                    and all(type(v) is int and 0 <= v < 65536 for v in words),
                    "sparse timestamp words",
                )
                start, end = [
                    sum(words[base + i] << (16 * i) for i in range(3))
                    for base in (0, 3)
                ]
                check(0 < end - start < 1 << 32, "sparse local interval")
                intervals.append(end - start)
        interval = max(intervals)
        check(0 < interval < 1 << 32, "sparse max-local interval")
        timings.append(interval)
    report = dict(
        passed=True,
        fixed_accuracy_passed=True,
        actors=s["rows"] * s["cols"],
        epochs=len(batches),
        output_values=manifest["expected_output_values"],
        factor_checks=checks,
        max_local_cycles=timings,
        timing_scope="Maximum local SDK interval; includes command launch between tic/toc, excludes host input/output copies; no synchronized global latency or hardware claim",
        partial_scope="first/last occupied original row per matrix partition before east/west reduction; sampled, not full trajectory",
        source_sha256=manifest["source_sha256"],
    )
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
