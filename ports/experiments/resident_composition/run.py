"""Execute a combined library dependency probe derived from a fresh HLS build.

The original HLS bundle is unchanged. The separate device adapter is explicitly
experimental and hash-recorded; this is not a claim of a CG frontend lowering.
"""

import argparse, datetime, hashlib, json, os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "toolchain"))
sys.path.insert(0, str(ROOT))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def worker(root):
    import math, numpy as np

    sys.path.insert(0, str(root / "implementation"))
    sys.path.insert(0, str(root))
    import spmv_host

    manifest = json.loads((root / "composition-manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert sha(root / name) == digest, name
    spmv_host.run(root)
    base = spmv_host.audit(
        root, json.loads((root / "original-build-manifest.json").read_text())
    )
    s = json.loads((root / "schedule.json").read_text())
    m = json.loads((root / "semantic.json").read_text())
    b = json.loads((root / "batches.json").read_text())
    r = json.loads((root / "results.json").read_text())
    from sparse_storage import distribute_x

    for epoch, (batch, output, d) in enumerate(zip(b, r["cases"], r["diagnostics"])):
        tiles = np.asarray(
            distribute_x(
                batch[m["nodes"][3]["host"]], s["M"], s["N"], s["rows"], s["cols"]
            ),
            np.float32,
        )
        np.testing.assert_array_equal(
            np.asarray(d["roundtrip"], np.float32).view(np.uint32),
            tiles.view(np.uint32),
        )
        np.testing.assert_array_equal(
            np.asarray(d["transpose"], np.float32).view(np.uint32),
            tiles.transpose(1, 0, 2).copy().view(np.uint32),
        )
        np.testing.assert_array_equal(np.asarray(d["queue_witness"]) & 252, 252)
        progress = np.asarray(d["composition_progress"])
        np.testing.assert_array_equal(
            progress,
            np.broadcast_to(
                [2 * (epoch + 1), epoch + 1, epoch + 1, epoch + 1], progress.shape
            ),
        )
        expected = math.fsum(float(v) ** 2 for v in output[m["nodes"][-1]["host"]])
        np.testing.assert_allclose(d["composition_result"], expected, rtol=3e-5, atol=0)
    (root / "composition-audit.json").write_text(
        json.dumps(
            dict(
                passed=True,
                epochs=len(b),
                original_spmv_audit=base,
                exact_transpose_and_roundtrip=True,
                all_pe_progress=True,
                global_output_square_sum=True,
            ),
            indent=2,
        )
        + "\n"
    )
    print("COMPOSITION PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--large", action="store_true")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Explicit tighter capacity contract and dead-phase scratch reuse; input matrices unchanged",
    )
    parser.add_argument(
        "--capacity",
        default="608,384,384",
        help="Explicit NNZ,column,row capacity contract for compact candidate",
    )
    a = parser.parse_args()
    if a.worker:
        worker(a.worker.resolve())
        return
    import numpy as np
    from compile import build
    from fixtures import batches
    from sdk_process import run_sdk

    n, mesh = (4096, 8) if a.large else (512, 4)
    out = (
        ROOT
        / "evidence"
        / (
            "resident-composition-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    out.mkdir()
    data = batches(f"mesh_spmv:{n}:{n}:{8*n}")
    for epoch, b in enumerate(data):
        x = np.arange(1, n + 1, dtype=np.float32) / n
        if epoch == 1:
            x[:] = 0
        elif epoch == 2:
            x[::2] *= -1
        elif epoch == 3:
            x = x[::-1].copy()
        b["x"] = x.tolist()
    source = ROOT / f"projects/sdk_examples/mesh_spmv_{n}x{n}_{mesh}x{mesh}/hls.cpp"
    if a.compact:
        assert a.large, "compact capacity candidate targets the large profile"
        nz, columns, rows = map(int, a.capacity.split(","))
        assert min(nz, columns, rows) > 0
        text = source.read_text().replace(
            "nnz_per_pe=1024 cols_per_pe=512 rows_per_pe=512",
            f"nnz_per_pe={nz} cols_per_pe={columns} rows_per_pe={rows}",
        )
        assert text != source.read_text()
        source = out / "bounded-source.cpp"
        source.write_text(text)
    build(
        source,
        out / "hls",
        epochs=4,
        bound=64,
        batches=data,
        sdk_options={"suppress_trace": True, "num_threads": 16, "dump_core": True},
    )
    device = out / "device"
    shutil.copytree(out / "hls", device)
    (device / "manifest.json").rename(device / "original-build-manifest.json")
    for target, original in (
        ("scalar_allreduce.csl", "scalar_allreduce.csl"),
        ("blas.csl", "sdk_blas.csl"),
    ):
        shutil.copy2(ROOT / "toolchain/runtime" / original, device / target)
    layout = (device / "layout.csl").read_text()
    layout = layout.replace(
        "layout {",
        'const c2d = @import_module("<collectives_2d/params>");\nlayout {',
        1,
    )
    layout = layout.replace(
        ".spmvParams = spmvParams,",
        ".spmvParams = spmvParams,\n.c2dParams=c2d.get_params(pcol_id,prow_id,.{.x_colors=.{@get_color(0),@get_color(7)},.x_entrypoints=.{@get_local_task_id(8),@get_local_task_id(9)},.y_colors=.{@get_color(8),@get_color(9)},.y_entrypoints=.{@get_local_task_id(22),@get_local_task_id(23)}}),",
    )
    end = layout.rindex("}")
    layout = (
        layout[:end]
        + '\n@export_name("queue_witness",[*]u16,true);@export_name("composition_progress",[*]u16,true);@export_name("composition_result",[*]f32,true);@export_name("transpose_buffer",[*]f32,true);\n'
        + layout[end:]
    )
    (device / "layout.csl").write_text(layout)
    kernel = (
        (device / "kernel.csl")
        .read_text()
        .replace(
            ".f_callback = sys_mod.unblock_cmd_stream,",
            ".f_callback = composition_done, .initialize_queues = false,",
        )
    )
    kernel = kernel.replace(
        "fn f_spmv() void {\n    spmv_mod.spmv(&x_tx_buf, &y_local_buf);\n}",
        "fn f_spmv() void { begin_composition(); }",
    )
    kernel += (Path(__file__).parent / "controller.csl").read_text()
    if a.compact:
        kernel = kernel.replace(
            ".vector_length=local_vec_sz,",
            ".vector_length=local_vec_sz, .transpose_chunk_length=32,",
        )
        kernel = kernel.replace(
            "var transpose_scratch = @zeros([spmvParams.pcols * local_vec_sz]u32);",
            "fn borrowed_scratch() *[spmvParams.pcols * 32]u32 { @comptime_assert(spmvParams.pcols * 32 <= max_local_nnz_rows); return @ptrcast(*[spmvParams.pcols * 32]u32, &spmv_mod.y_vals_north_buf); }",
        )
        kernel = kernel.replace("&transpose_scratch)", "borrowed_scratch())")
    (device / "kernel.csl").write_text(kernel)
    host = (device / "implementation/mesh_spmv_sdk.py").read_text()
    host = host.replace(
        "    runner.load()",
        '    ids.update({name:runner.get_id(name) for name in ("composition_progress","composition_result","transpose_buffer","queue_witness")})\n    runner.load()',
        1,
    )
    host = host.replace(
        '                    timing=get("time_buf_u16", 6, True),',
        """                    timing=get("time_buf_u16", 6, True),
                    queue_witness=get("queue_witness",6,True),
                    composition_progress=get("composition_progress",4,True),
                    composition_result=get("composition_result",1),
                    transpose=get("transpose_buffer",s["geometry"]["local_vec_sz"]),
                    roundtrip=get("x_tx_buf",s["geometry"]["local_vec_sz"]),""",
    )
    start = host.index(
        "    with tempfile.TemporaryDirectory() as tmp:", host.index("def audit(")
    )
    end = host.index("    batches =", start)
    host = host[:start] + """    import hashlib
    for name,digest in read(root,"composition-manifest.json")["files"].items():
        check(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,"composition source hash "+name)
""" + host[end:]
    (device / "spmv_host.py").write_text(host)
    shutil.copy2(__file__, device / "driver.py")
    manifest = dict(
        scope="experimental combined adapter from preserved HLS SpMV; not CG",
        source_manifest_sha256=sha(out / "hls/manifest.json"),
        files={p.name: sha(p) for p in device.iterdir() if p.is_file()},
        target="SDK2.10.1 WSE3 default memcpy context",
        resources={
            "spmv_tasks": list(range(11, 21)) + [24, 25, 26],
            "collective_tasks": [8, 9, 22, 23],
            "callback_task": 10,
            "collective_input_queues": [3, 5],
            "collective_output_queues": [6, 7],
            "collective_colors": [0, 7, 8, 9],
        },
    )
    (device / "composition-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(out, flush=True)
    with (device / "sdk.log").open("w") as log:
        run_sdk(
            [
                "/home/lingzhi/cerebras/sdk/2.10.1/cs_python",
                str(device / "driver.py"),
                "--worker",
                str(device),
            ],
            device,
            dict(
                os.environ,
                SINGULARITYENV_CS_TARGET="SDR",
                SINGULARITYENV_PYTHONUNBUFFERED="1",
            ),
            log,
            600,
        )
    print("COMPOSITION PASS", flush=True)


if __name__ == "__main__":
    main()
