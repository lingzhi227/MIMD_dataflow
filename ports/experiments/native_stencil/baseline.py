"""Resident SDK-library baseline, independent of the production HLS backend.

Copies unchanged SDK libraries; adapts only the benchmark application wrapper.
Run with the ports virtualenv. The worker runs inside SDK 2.10.1.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

PORTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PORTS / "toolchain"))


def read(root, name):
    return json.loads((root / name).read_text())


def save(root, name, value):
    (root / name).write_text(json.dumps(value, indent=2) + "\n")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("upstream adapter anchor changed: " + old[:80])
    return text.replace(old, new)


def prepare(source, dest, block):
    from library_contracts import sdk_stencil
    from grid_plan import plan

    m = read(source, "semantic.json")
    s = read(source, "schedule.json")
    if s != plan(m):
        raise ValueError("source schedule does not match checked HLS semantics")
    contract = sdk_stencil(m, block, allow_reassociation=True)
    dest.mkdir(parents=True, exist_ok=False)
    save(dest, "library-contract.json", contract)
    upstream = PORTS / "projects/sdk_examples/upstream/benchmarks"
    lib = dest / "benchmark-libs"
    for name in ("allreduce", "stencil_3d_7pts"):
        shutil.copytree(upstream / "benchmark-libs" / name, lib / name)
    app = dest / "app/src"
    shutil.copytree(upstream / "7pt-stencil-spmv/src", app)
    original = {
        str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in dest.rglob("*.csl")
    }
    for name in ("semantic.json", "schedule.json", "batches.json", "hls.cpp"):
        if (source / name).exists():
            shutil.copyfile(source / name, dest / name)
    shutil.copytree(
        PORTS / "toolchain",
        dest / "toolchain",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copyfile(__file__, dest / "baseline.py")
    steps = s["grid"]["steps"]
    kernel = (app / "kernel.csl").read_text()
    anchor = ".stencil_params = stencilParams,\n     .f_callback = sys_mod.unblock_cmd_stream,"
    kernel = replace_once(
        kernel,
        anchor,
        ".stencil_params = stencilParams,\n     .f_callback = stencil_finished,",
    )
    kernel = replace_once(
        kernel,
        "    stencil_mod.spmv(n, &stencil_coeff, &x, &y);",
        "    @assert(n == MAX_ZDIM); step = 0;\n    stencil_mod.spmv(n, &stencil_coeff, &x, &y);",
    )
    kernel += f"""
// HLS baseline adapter: library sources above are unchanged.
const STEPS:i16={steps};
var step:i16=0;
var history=@zeros([MAX_ZDIM*STEPS]f32);
var ptr_history:[*]f32=&history;
const x_dsd=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{MAX_ZDIM}}->x[i]}});
const y_dsd=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{MAX_ZDIM}}->y[i]}});
const history_dsd=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{MAX_ZDIM}}->history[i]}});
fn stencil_finished() void {{
    const at=@increment_dsd_offset(history_dsd,@as(i16,step)*MAX_ZDIM,f32);
    @fmovs(at,y_dsd);
    @fmovs(x_dsd,y_dsd);
    step+=1;
    if(step<STEPS){{stencil_mod.spmv(MAX_ZDIM,&stencil_coeff,&x,&y);}}
    else{{sys_mod.unblock_cmd_stream();}}
}}
comptime {{@export_symbol(ptr_history,"history");}}
"""
    (app / "kernel.csl").write_text(kernel)
    layout = (app / "layout.csl").read_text()
    layout = replace_once(
        layout,
        '@export_name("y", [*]f32, true);',
        '@export_name("y", [*]f32, true);\n    @export_name("history", [*]f32, true);',
    )
    (app / "layout.csl").write_text(layout)
    save(
        dest,
        "manifest.json",
        {
            "source_run": str(source),
            "block_size": block,
            "upstream_commit": "4866cf330333446cb5e529e10f36be4600d1df29",
            "upstream_sha256": original,
            "snapshot_sha256": {
                str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in dest.rglob("*")
                if p.is_file()
            },
            "numeric_policy": "SDK ordered FMA, compare every step with HLS f32 semantics using rtol=3e-5, atol=3e-6",
            "mapping": "HLS [x,y,z] to SDK [y,x,z]; swap south/north coefficients (indices 2,3)",
            "timing_scope": "Native contiguous layout and memcpy differ from HLS spaced layout and column streaming. Total cycles are not a controlled speedup comparison.",
        },
    )


def worker(root):
    import numpy as np
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime,
        MemcpyDataType,
        MemcpyOrder,
    )

    s, m = read(root, "schedule.json"), read(root, "semantic.json")
    g = s["grid"]
    nx, ny, z, steps = (g[k] for k in ("x", "y", "z", "steps"))
    block = read(root, "manifest.json")["block_size"]
    os.chdir(root / "app")
    command = [
        "cslc",
        "src/layout.csl",
        "--arch=wse3",
        f"--fabric-dims={nx+7},{ny+2}",
        "--fabric-offsets=4,1",
        f"--params=width:{nx},height:{ny},MAX_ZDIM:{z},BLOCK_SIZE:{block}",
        *[f"--params=C{i}_ID:{i}" for i in range(9)],
        "-o=out",
        "--memcpy",
        "--channels=1",
        "--width-west-buf=0",
        "--width-east-buf=0",
    ]
    save(root, "compiler-command.json", command)
    subprocess.run(command, check=True)
    runner = SdkRuntime("out")
    symbols = {
        name: runner.get_id(name) for name in ("x", "y", "stencil_coeff", "history")
    }
    runner.load()
    runner.run()
    result = {"success": False, "epochs": []}

    def h2d(name, array):
        runner.memcpy_h2d(
            symbols[name],
            np.asarray(array, np.float32).ravel(order="F"),
            0,
            0,
            nx,
            ny,
            array.shape[2],
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.COL_MAJOR,
            nonblock=False,
        )

    def d2h(name, length):
        buf = np.zeros(nx * ny * length, np.float32)
        runner.memcpy_d2h(
            buf,
            symbols[name],
            0,
            0,
            nx,
            ny,
            length,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.COL_MAJOR,
            nonblock=False,
        )
        return buf.reshape(ny, nx, length, order="F").transpose(1, 0, 2)

    try:
        for epoch, batch in enumerate(read(root, "batches.json")):
            field = np.array(batch[m["nodes"][0]["host"]], np.float32).reshape(
                nx, ny, z
            )
            coeff = np.array(batch[m["nodes"][1]["host"]], np.float32)[
                read(root, "library-contract.json")["coefficient_permutation"]
            ]
            h2d("x", field.transpose(1, 0, 2))
            h2d("stencil_coeff", np.broadcast_to(coeff, (ny, nx, 7)))
            runner.launch("f_spmv", np.int16(z), nonblock=False)
            result["epochs"].append(
                {
                    "final": d2h("y", z).ravel().tolist(),
                    "history": d2h("history", z * steps)
                    .reshape(nx, ny, steps, z)
                    .tolist(),
                }
            )
            save(root, "results.json", result)
            print("NATIVE RESIDENT EPOCH", epoch + 1, "COMPLETE", flush=True)
    finally:
        runner.stop()
    result["success"] = True
    save(root, "results.json", result)


def audit(root):
    import numpy as np
    from grid_ir import simulate

    manifest = read(root, "manifest.json")
    for path, digest in manifest["snapshot_sha256"].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise ValueError("baseline snapshot changed: " + path)
    for path, digest in manifest["upstream_sha256"].items():
        if (
            path.startswith("benchmark-libs/")
            and hashlib.sha256((root / path).read_bytes()).hexdigest() != digest
        ):
            raise ValueError("native library modified: " + path)
    m, s, result = (
        read(root, "semantic.json"),
        read(root, "schedule.json"),
        read(root, "results.json"),
    )
    batches = read(root, "batches.json")
    if not result["success"] or len(result["epochs"]) != len(batches):
        raise ValueError("incomplete native baseline")
    maximum = 0.0
    count = 0
    for batch, actual in zip(batches, result["epochs"]):
        final, traces = simulate(m, batch, True)
        np.testing.assert_allclose(actual["final"], final, rtol=3e-5, atol=3e-6)
        for node in s["nodes"]:
            x, y = node["tile"]
            observed = np.asarray(actual["history"][x][y]).ravel()
            expected = np.asarray(traces[node["id"]])
            np.testing.assert_allclose(observed, expected, rtol=3e-5, atol=3e-6)
            maximum = max(maximum, float(np.max(np.abs(observed - expected))))
            count += len(observed)
    report = {
        "passed": True,
        "grid": s["grid"],
        "epochs": len(batches),
        "step_values_checked": count,
        "max_abs_error_vs_hls_f32": maximum,
        "libraries_unchanged": True,
        "timing_scope": manifest["timing_scope"],
    }
    stats = root / "app/sim_stats.json"
    if stats.exists():
        report["total_simulator_cycles"] = json.loads(stats.read_text())["cycle_count"]
    save(root, "audit.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--block", type=int, default=32)
    p.add_argument("--sdk", default="/home/lingzhi/cerebras/sdk/2.10.1/cs_python")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--audit", action="store_true")
    a = p.parse_args()
    if a.worker:
        return worker(a.source.resolve())
    if a.audit:
        sys.path.insert(0, str(a.source.resolve() / "toolchain"))
        print(json.dumps(audit(a.source.resolve()), indent=2))
        return
    root = (
        PORTS
        / "evidence"
        / ("native-stencil-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    prepare(a.source.resolve(), root, a.block)
    manifest = read(root, "manifest.json")
    sdk = Path(a.sdk).resolve()
    images = list(sdk.parent.glob("sdk-cbcore-2.10.1-*.sif"))
    if len(images) != 1:
        raise ValueError("expected exactly one SDK 2.10.1 image next to wrapper")
    with images[0].open("rb") as stream:
        image_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest["sdk"] = {
        "wrapper": str(sdk),
        "wrapper_sha256": hashlib.sha256(sdk.read_bytes()).hexdigest(),
        "image": str(images[0]),
        "image_sha256": image_hash,
    }
    save(root, "manifest.json", manifest)
    print(root, flush=True)
    from sdk_process import run_sdk

    env = dict(
        os.environ, SINGULARITYENV_CS_TARGET="SDR", SINGULARITYENV_PYTHONUNBUFFERED="1"
    )
    with (root / "sdk.log").open("w") as log:
        run_sdk(
            [a.sdk, str(root / "baseline.py"), str(root), "--worker"],
            root / "app",
            env,
            log,
            a.timeout,
        )
    print(json.dumps(audit(root), indent=2))


if __name__ == "__main__":
    main()
