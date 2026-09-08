"""SDK transport for resident half-input regions; no application arithmetic.

Packing and final decoding are supplied by the lowering. Raw observer widths
are explicit, while protocol/numerical auditing remains in separate modules.
"""

import json, time
from pathlib import Path
import numpy as np
from frontend import check


def read(root, name):
    return json.loads((Path(root) / name).read_text())


def save(root, name, value):
    p = Path(root) / name
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(value) + "\n")
    tmp.replace(p)


def run(root, parameters, extents, pack_batch, decode, word_bits=None):
    import os, subprocess
    from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder
    from cerebras.sdk.sdk_utils import input_array_to_u32
    from mesh_common import sdk_runtime

    root = Path(root).resolve()
    os.chdir(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    rows, cols = s["rows"], s["cols"]
    ports = extents(s)
    widths = word_bits or {}
    check(
        all(k in ports and v in (16, 32) for k, v in widths.items()),
        "raw observer word widths",
    )

    def stage(operation, epoch=None, port=None):
        save(
            root,
            "runtime-stage.json",
            dict(operation=operation, epoch=epoch, port=port),
        )

    parameter_argument = parameters(s)
    check(
        parameter_argument is None or isinstance(parameter_argument, str),
        "CSL parameter argument",
    )
    cmd = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        f"--fabric-dims={cols+7},{rows+2}",
        "--fabric-offsets=4,1",
        *([] if parameter_argument is None else [parameter_argument]),
        "-o=out",
        "--memcpy",
        "--channels=1",
    ]
    save(root, "sdk-command.json", cmd)
    stage("compiling")
    subprocess.run(cmd, check=True)
    runner = sdk_runtime(root)
    ids = {k: runner.get_id(k) for k in ports}
    stage("loading")
    runner.load()
    stage("starting")
    runner.run()
    result = dict(
        success=False,
        runtime_instances=1,
        cases=[],
        diagnostics=[],
        launches=[],
        host_call_seconds=[],
        host_timing_scope="Host full input/launch/output call, excluding compile/load; not device throughput",
    )
    try:
        stage("initialize")
        runner.launch("init_task", nonblock=False)
        for epoch, batch in enumerate(read(root, "batches.json")):
            started = time.monotonic()
            for name, a in pack_batch(s, m, batch).items():
                a = np.asarray(a, np.float16)
                check(
                    name in ports and a.shape == (rows, cols, ports[name]),
                    "packed input extent",
                )
                check(np.all(np.isfinite(a)), "finite half transport input")
                stage("host_to_device", epoch, name)
                runner.memcpy_h2d(
                    ids[name],
                    input_array_to_u32(a.ravel(), 1, 1),
                    0,
                    0,
                    cols,
                    rows,
                    ports[name],
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_16BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
            stage("launch", epoch)
            runner.launch("hls_main", nonblock=False)
            result["launches"].append("hls_main")
            diagnostics = {}
            for name, n in ports.items():
                stage("device_to_host", epoch, name)
                raw = np.zeros(rows * cols * n, np.uint32)
                runner.memcpy_d2h(
                    raw,
                    ids[name],
                    0,
                    0,
                    cols,
                    rows,
                    n,
                    streaming=False,
                    data_type=(
                        MemcpyDataType.MEMCPY_32BIT
                        if widths.get(name, 16) == 32
                        else MemcpyDataType.MEMCPY_16BIT
                    ),
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
                diagnostics[name] = (
                    raw.astype(np.uint32 if widths.get(name, 16) == 32 else np.uint16)
                    .reshape(rows, cols, n)
                    .tolist()
                )
            result["cases"].append(decode(s, m, diagnostics))
            result["diagnostics"].append(diagnostics)
            result["host_call_seconds"].append(time.monotonic() - started)
            save(root, "results.json", result)
            print("RESIDENT REGION", epoch + 1, flush=True)
    finally:
        stage("stopping")
        runner.stop()
    result["success"] = True
    save(root, "results.json", result)
    stage("completed")
