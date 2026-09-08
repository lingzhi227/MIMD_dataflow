"""Instrument a pinned native factorization baseline; compare the first input.

Cholesky uses the original SDK source; LU/QR use explicitly identified prior
SDK 2.10.1 single-RX/directed migrations. These baselines are not HLS ports.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = repository_root(__file__)


def read(p):
    return json.loads(Path(p).read_text())


def worker(root):
    import numpy as np
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime,
        SimfabConfig,
        SdkTarget,
        get_platform,
        MemcpyOrder,
        MemcpyDataType,
    )

    os.chdir(root)
    s = read("schedule.json")
    rows, cols, nt = s.get("rows", s.get("P")), s.get("cols", s.get("P")), s["Nt"]
    subprocess.run(read("sdk-command.json"), check=True)
    runner = SdkRuntime(
        "out",
        get_platform(
            None, SimfabConfig(**read("runtime-options.json")), SdkTarget.WSE3
        ),
    )
    tile_id, time_id = runner.get_id(
        "A" if s["profile"] in ("mesh_lu.v1", "mesh_qr.v1") else "tile"
    ), runner.get_id("timing")
    a = np.asarray(read("input.json"), np.float32).reshape(rows, nt, cols, nt)
    packed = a.transpose(0, 2, 1, 3).reshape(rows, cols, nt * nt).ravel(order="F")
    runner.load()
    runner.run()
    try:
        runner.memcpy_h2d(
            tile_id,
            packed,
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
        runner.launch(
            "start" if s["profile"] in ("mesh_lu.v1", "mesh_qr.v1") else "f_chol",
            nonblock=False,
        )
        raw = np.zeros(rows * cols * nt * nt, np.float32)
        runner.memcpy_d2h(
            raw,
            tile_id,
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
        timing = np.zeros(rows * cols * 6, np.uint32)
        runner.memcpy_d2h(
            timing,
            time_id,
            0,
            0,
            cols,
            rows,
            6,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_16BIT,
            order=MemcpyOrder.COL_MAJOR,
            nonblock=False,
        )
        matrix = (
            raw.reshape(rows, cols, nt * nt, order="F")
            .reshape(rows, cols, nt, nt)
            .transpose(0, 2, 1, 3)
            .reshape(rows * nt, cols * nt)
        )
        Path("results.json").write_text(
            json.dumps(
                {
                    "raw_tile_matrix": matrix.tolist(),
                    "timing": timing.reshape(rows, cols, 6, order="F").tolist(),
                }
            )
            + "\n"
        )
    finally:
        runner.stop()


def main(bundle):
    import numpy as np

    sys.path.insert(0, str(ROOT / "lib"))
    from sdk_process import run_sdk
    from mesh_cholesky_sdk import check_factor

    bundle = Path(bundle).resolve()
    subprocess.run(
        [sys.executable, str(bundle / "implementation/validate.py"), str(bundle)],
        check=True,
        capture_output=True,
    )
    s = read(bundle / "schedule.json")
    if s["profile"] not in ("mesh_cholesky.v1", "mesh_lu.v1", "mesh_qr.v1"):
        raise ValueError("expected validated factorization bundle")
    is_lu = s["profile"] == "mesh_lu.v1"
    is_qr = s["profile"] == "mesh_qr.v1"
    if is_lu:
        from mesh_lu_sdk import check_factor
    elif is_qr:
        from mesh_qr_sdk import check_factor
    out = (
        ROOT
        / "validation/evidence"
        / (
            ("native-qr-" if is_qr else "native-lu-" if is_lu else "native-cholesky-")
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    out.mkdir()
    upstream = (
        ROOT
        / (
            "experiments/reference/matrix-qr-sdk2101"
            if is_qr
            else "experiments/reference/matrix-lu-sdk2101"
        )
        if is_lu or is_qr
        else ROOT / "third_party/sources/sdk_examples/benchmarks/cholesky"
    )
    originals = {
        name: (upstream / name).read_text()
        for name in (
            ("pe_program.csl", "layout.csl")
            if is_lu or is_qr
            else ("pe.csl", "layout.csl", "launch.csl")
        )
    }
    # Start/stop positions match HLS factor timing; prepare and I/O are excluded.
    declarations = """
const timestamp = @import_module("<time>");
var started = @zeros([3]u16);
var ended = @zeros([3]u16);
var timing = @zeros([6]u16);
var ptr_timing: [*]u16 = &timing;
"""
    if is_lu or is_qr:
        pe = originals["pe_program.csl"].replace(
            "const sys_mod =", declarations + "\nconst sys_mod ="
        )
        pe = pe.replace(
            "fn start() void {",
            "fn start() void {\n  timestamp.enable_tsc();\n  timestamp.get_timestamp(&started);",
        )
        pe = pe.replace(
            "sys_mod.unblock_cmd_stream();",
            "timestamp.get_timestamp(&ended);\n  for (@range(u16,3)) |i| { timing[i] = started[i]; timing[i+3] = ended[i]; }\n  sys_mod.unblock_cmd_stream();",
        )
        pe = pe.replace(
            "  @export_symbol(start);",
            '  @export_symbol(start);\n  @export_symbol(ptr_timing, "timing");',
        )
        layout = (
            originals["layout.csl"]
            .replace('"pe_program.csl"', '"pe.csl"')
            .replace(
                '  @export_name("A",',
                '  @export_name("timing", [*]u16, true);\n  @export_name("A",',
            )
        )
        (out / "pe.csl").write_text(pe)
        (out / "layout.csl").write_text(layout)
    else:
        pe = originals["pe.csl"].replace(
            "const math =", declarations + "\nconst math ="
        )
        pe = pe.replace(
            "fn f_chol() void {",
            "fn f_chol() void {\n  timestamp.enable_tsc();\n  timestamp.get_timestamp(&started);",
        )
        pe = pe.replace(
            "    sys_mod.unblock_cmd_stream();",
            """    timestamp.get_timestamp(&ended);
        for (@range(u16,3)) |i| { timing[i] = started[i]; timing[i+3] = ended[i]; }
        sys_mod.unblock_cmd_stream();""",
        )
        pe = pe.replace(
            "  @export_symbol(f_chol);",
            '  @export_symbol(f_chol);\n  @export_symbol(ptr_timing, "timing");',
        )
        launch = originals["launch.csl"].replace(
            "fn f_chol()", declarations + "\nfn f_chol()"
        )
        launch = launch.replace(
            "  @export_symbol(f_chol);",
            '  @export_symbol(f_chol);\n  @export_symbol(ptr_timing, "timing");',
        )
        layout = originals["layout.csl"].replace(
            '  @export_name("tile",',
            '  @export_name("timing", [*]u16, true);\n  @export_name("tile",',
        )
        for name, contents in (
            ("pe.csl", pe),
            ("layout.csl", layout),
            ("launch.csl", launch),
        ):
            (out / name).write_text(contents)
    for name in ("schedule.json", "sdk-command.json", "runtime-options.json"):
        shutil.copyfile(bundle / name, out / name)
    input_host = read(bundle / "semantic.json")["nodes"][0]["host"]
    a = read(bundle / "batches.json")[0][input_host]
    (out / "input.json").write_text(json.dumps(a) + "\n")
    shutil.copyfile(__file__, out / "driver.py")
    # Preserve the exact numerical checker used by this native baseline.
    checker = (
        "mesh_qr_sdk.py"
        if is_qr
        else "mesh_lu_sdk.py" if is_lu else "mesh_cholesky_sdk.py"
    )
    shutil.copyfile(ROOT / "lib" / checker, out / "numerical_checker.py")
    shutil.copyfile(ROOT / "lib/Frontend/frontend.py", out / "frontend.py")
    sdk_image = Path(
        "/home/lingzhi/cerebras/sdk/2.10.1/sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif"
    )
    image_hash = hashlib.file_digest(sdk_image.open("rb"), "sha256").hexdigest()
    if image_hash != "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d":
        raise ValueError("SDK image hash mismatch")
    manifest = {
        "sdk_sha256": image_hash,
        "oracle_numpy": np.__version__,
        "profile": s["profile"],
        "scope_of_source_hashes": "Baseline source before timestamp instrumentation; migrated source for Matrix LU/QR",
        "kind": (
            "timestamp-only SDK2.10.1 migrated Matrix "
            + ("QR" if is_qr else "LU")
            + " baseline, one cold invocation"
            if is_lu or is_qr
            else "instrumented original SDK baseline, one cold invocation"
        ),
        "upstream_commit": (
            "016156e79b63fe45e118580da8db694285b6c6d9"
            if is_lu or is_qr
            else "4866cf330333446cb5e529e10f36be4600d1df29"
        ),
        "original_sha256": {
            name: hashlib.sha256(value.encode()).hexdigest()
            for name, value in originals.items()
        },
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
        },
        "comparison_bundle": str(bundle),
        "comparison_results_sha256": hashlib.sha256(
            (bundle / "results.json").read_bytes()
        ).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out, flush=True)
    with (out / "sdk.log").open("w") as log:
        run_sdk(
            [
                "/home/lingzhi/cerebras/sdk/2.10.1/cs_python",
                str(out / "driver.py"),
                "--worker",
                str(out),
            ],
            out,
            dict(
                os.environ,
                SINGULARITYENV_CS_TARGET="SDR",
                SINGULARITYENV_PYTHONUNBUFFERED="1",
            ),
            log,
            900,
        )
    result = read(out / "results.json")
    factor = np.asarray(result["raw_tile_matrix"])
    if not (is_lu or is_qr):
        factor = np.tril(factor)
    checks = check_factor(np.asarray(a).reshape(factor.shape), factor)
    hls = read(bundle / "results.json")
    values = np.asarray(next(iter(hls["cases"][0].values()))).reshape(factor.shape)
    np.testing.assert_allclose(values, factor, rtol=3e-5, atol=3e-6)

    def durations(timing):
        return [
            (
                sum(int(timing[y][x][i + 3]) << (16 * i) for i in range(3))
                - sum(int(timing[y][x][i]) << (16 * i) for i in range(3))
            )
            % (1 << 48)
            for y in range(s.get("rows", s.get("P")))
            for x in range(s.get("cols", s.get("P")) if is_lu or is_qr else y + 1)
        ]

    native_ticks = durations(result["timing"])
    hls_ticks = durations(hls["diagnostics"][0]["timing"])
    if any(not 0 < v < 1 << 32 for v in native_ticks + hls_ticks):
        raise ValueError("baseline timestamp bounds")
    report = {
        "passed": True,
        "numerical_checks": checks,
        "same_input_epoch": 0,
        "native_per_pe_cycles": native_ticks,
        "hls_per_pe_cycles": hls_ticks,
        "hls_over_native_max_pe_cycles": max(hls_ticks) / max(native_ticks),
        "scope": "First factor invocation, matching per-PE timing boundaries and SDK options. Baseline (see manifest for original or migrated provenance) adds timestamps only; HLS includes corner checkpoints and warm preparation. Excludes prepare/host I/O, no hardware claim.",
    }
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle")
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    worker(Path(a.bundle).resolve()) if a.worker else main(a.bundle)
