"""Shared SDK compile/transport loop; all numerical iteration remains in CSL."""

import json
from pathlib import Path
from mesh_spmv_sdk import read
from resident_abi import schema


def run(root):
    import os, subprocess, numpy as np
    from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder
    from mesh_common import sdk_runtime

    root = Path(root).resolve()
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    abi = read(root, "host-abi.json")
    assert abi == schema(s), "resident ABI regeneration"
    rows, cols = s["rows"], s["cols"]
    local = s["geometry"]["local_vec_sz"]
    params = dict(
        prows=rows,
        pcols=cols,
        nrows=s["M"],
        ncols=s["N"],
        max_local_nnz=s["capacity"]["nnz"],
        max_local_nnz_cols=s["capacity"]["columns"],
        max_local_nnz_rows=s["capacity"]["rows"],
        max_iterations=s["max_iterations"],
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
    runner = sdk_runtime(root)
    ids = {
        name: runner.get_id(name) for name in list(s["extents"]) + list(abi["symbols"])
    }

    def transfer(name, data, kind, receive=False):
        data = np.asarray(data, dtype=np.float32 if kind == "f32" else np.uint32)
        length = data.shape[2]
        flat = data.ravel(order="F").copy()
        fn = runner.memcpy_d2h if receive else runner.memcpy_h2d
        args = (flat, ids[name]) if receive else (ids[name], flat)
        fn(
            *args,
            0,
            0,
            cols,
            rows,
            length,
            streaming=False,
            data_type=(
                MemcpyDataType.MEMCPY_16BIT
                if kind == "u16"
                else MemcpyDataType.MEMCPY_32BIT
            ),
            order=MemcpyOrder.COL_MAJOR,
            nonblock=False,
        )
        return flat.reshape(rows, cols, length, order="F").tolist()

    result = dict(
        success=False, cases=[], diagnostics=[], runtime_instances=1, launches=[]
    )

    def save():
        (root / "results.json").write_text(json.dumps(result) + "\n")

    runner.load()
    runner.run()
    try:
        for epoch, (b, p) in enumerate(
            zip(read(root, "batches.json"), read(root, "sparse-packing.json"))
        ):
            for name in s["extents"]:
                transfer(
                    name,
                    [[t[name] for t in row] for row in p["tiles"]],
                    "f32" if name == "mat_vals_buf" else "u16",
                )
            for symbol, key, kind in abi["packed_inputs"]:
                transfer(symbol, p[key], kind)
            for symbol, index in abi["vector_inputs"]:
                transfer(
                    symbol,
                    np.asarray(b[m["nodes"][index]["host"]]).reshape(rows, cols, local),
                    "f32",
                )
            for symbol, index, kind in abi["replicated_inputs"]:
                transfer(
                    symbol, np.tile(b[m["nodes"][index]["host"]], (rows, cols, 1)), kind
                )
            (root / "runtime-stage.json").write_text(
                json.dumps(dict(epoch=epoch, operation="launch " + abi["launch"]))
            )
            runner.launch(abi["launch"], nonblock=False)
            result["launches"].append(abi["launch"])
            d = {
                name: transfer(
                    name,
                    np.zeros((rows, cols, abi["symbols"][name]["length"])),
                    abi["symbols"][name]["dtype"],
                    True,
                )
                for name in abi["read_symbols"]
            }
            # Retain established diagnostic names across existing solver evidence.
            for old, new in [
                ("hls_progress", "progress"),
                ("hls_partial", "partial"),
                ("y_local_buf", "final_ax"),
            ]:
                d[new] = d.pop(old)
            o = {}
            for n in m["nodes"][abi["output_node_start"] :]:
                field = n["inputs"][0].split(".")[1]
                v = d[abi["result_fields"][field]]
                o[n["host"]] = (
                    np.asarray(v).ravel().tolist()
                    if field == abi["vector_field"]
                    else v[0][0][: n["shape"][0] * n["shape"][1]]
                )
            result["cases"].append(o)
            result["diagnostics"].append(d)
            save()
            print(s["profile"], "EPOCH", epoch + 1, "COMPLETE", flush=True)
    finally:
        runner.stop()
    result["success"] = True
    save()
