"""Shared row-major resident factorization host transport and prepare lifecycle."""

import json
import os
from pathlib import Path
import subprocess


def read(root, name):
    return json.loads((Path(root) / name).read_text())


def run_factor(root, matrix_symbol, entrypoint, params, diagnostics=None):
    import numpy as np
    from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder, MemcpyDataType
    from mesh_common import pack_tiles, unpack_tiles, sdk_runtime

    root = Path(root).resolve()
    s, m = read(root, "schedule.json"), read(root, "semantic.json")
    rows, cols, nt = s.get("rows", s.get("P")), s.get("cols", s.get("P")), s["Nt"]
    if diagnostics is None:
        diagnostics = [
            ("checkpoints", s["N"] * 2, False, [s["N"], 2]),
            ("timing", 6, True, [6]),
            ("progress", 2, True, [2]),
        ]
    os.chdir(root)
    cmd = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        f"--fabric-dims={cols+7},{rows+2}",
        "--fabric-offsets=4,1",
        "--params=" + ",".join(f"{name}:{s[key]}" for name, key in params.items()),
        "-o=out",
        "--memcpy",
        "--channels=1",
        "--width-west-buf=0",
        "--width-east-buf=0",
    ]
    (root / "sdk-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    subprocess.run(cmd, check=True)
    runner = sdk_runtime(root)
    ids = {
        key: runner.get_id(key) for key in [matrix_symbol] + [d[0] for d in diagnostics]
    }
    runner.load()
    runner.run()
    result = {
        "success": False,
        "cases": [],
        "diagnostics": [],
        "runtime_instances": 1,
        "prepare_barriers": 0,
    }

    def save():
        (root / "results.json").write_text(json.dumps(result) + "\n")

    def get(key, length, short=False):
        values = np.zeros(rows * cols * length, np.uint32 if short else np.float32)
        runner.memcpy_d2h(
            values,
            ids[key],
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
        return values.reshape(rows, cols, length, order="F")

    try:
        for epoch, batch in enumerate(read(root, "batches.json")):
            a = np.asarray(batch[m["nodes"][0]["host"]], np.float32).reshape(
                m["nodes"][0]["shape"]
            )
            tiles = pack_tiles(a, rows, cols, "C")
            runner.memcpy_h2d(
                ids[matrix_symbol],
                tiles.ravel(order="F"),
                0,
                0,
                cols,
                rows,
                nt * nt,
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.COL_MAJOR,
                nonblock=False,
            )
            runner.launch("prepare", nonblock=False)
            result["prepare_barriers"] += 1
            runner.launch(entrypoint, nonblock=False)
            factor = unpack_tiles(get(matrix_symbol, nt * nt), nt, nt, "C")
            result["cases"].append({m["nodes"][2]["host"]: factor.ravel().tolist()})
            result["diagnostics"].append(
                {
                    name: get(name, length, short).reshape(rows, cols, *shape).tolist()
                    for name, length, short, shape in diagnostics
                }
            )
            save()
            print(s["profile"], "EPOCH", epoch + 1, "COMPLETE", flush=True)
    finally:
        runner.stop()
    result["success"] = True
    save()
