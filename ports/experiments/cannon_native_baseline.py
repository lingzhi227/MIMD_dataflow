"""Pinned source Cannon with SDK2.10.1 ABI migration and timed warm reset."""

import argparse, datetime, hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worker(root):
    import numpy as np

    sys.path.insert(0, str(root / "implementation"))
    from mesh_common import sdk_runtime, unpack_tiles
    from mesh_cannon import pack
    from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder, MemcpyDataType

    for name, digest in read(root / "provenance.json")["files"].items():
        assert sha(root / name) == digest
    os.chdir(root)
    subprocess.run(read("sdk-command.json"), check=True)
    s = read("schedule.json")
    m = read("semantic.json")
    p = s["P"]
    t = s["Mt"]
    runner = sdk_runtime(root)
    ids = {k: runner.get_id(k) for k in ["A", "B", "C", "total_timing"]}
    runner.load()
    runner.run()
    out = dict(success=False, cases=[], timing=[], runtime_instances=1)
    try:
        for b in read("batches.json"):
            for key, node in zip(("A", "B"), m["nodes"][:2]):
                v = pack(
                    np.asarray(b[node["host"]], np.float32).reshape(node["shape"]),
                    p,
                    key,
                )
                runner.memcpy_h2d(
                    ids[key],
                    v.ravel(order="F"),
                    0,
                    0,
                    p,
                    p,
                    t * t,
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_32BIT,
                    order=MemcpyOrder.COL_MAJOR,
                    nonblock=False,
                )
            runner.launch("main", nonblock=False)
            v = np.zeros(p * p * t * t, np.float32)
            time = np.zeros(p * p * 6, np.uint32)
            for key, buf, length, typ in [
                ("C", v, t * t, MemcpyDataType.MEMCPY_32BIT),
                ("total_timing", time, 6, MemcpyDataType.MEMCPY_16BIT),
            ]:
                runner.memcpy_d2h(
                    buf,
                    ids[key],
                    0,
                    0,
                    p,
                    p,
                    length,
                    streaming=False,
                    data_type=typ,
                    order=MemcpyOrder.COL_MAJOR,
                    nonblock=False,
                )
            out["cases"].append(
                unpack_tiles(v.reshape(p, p, t * t, order="F"), t, t, "C").tolist()
            )
            out["timing"].append(time.reshape(p, p, 6, order="F").tolist())
            (root / "results.json").write_text(json.dumps(out) + "\n")
    finally:
        runner.stop()
    out["success"] = True
    (root / "results.json").write_text(json.dumps(out) + "\n")


def native_pe(source):
    s = source.replace(
        "param memcpy_params: comptime_struct;",
        "param memcpy_params;\nparam px:u16;param py:u16;",
    )
    s = re.sub(r"\.fabric_color = (?:send|recv)_[AB]_color, ", "", s)
    for bank in ("input", "output"):
        s = s.replace(f"@get_{bank}_queue(0)", f"@get_{bank}_queue(2)").replace(
            f"@get_{bank}_queue(1)", f"@get_{bank}_queue(3)"
        )
    s = (
        s.replace("layout_mod.get_x_coord()", "px")
        .replace("layout_mod.get_y_coord()", "py")
        .replace("@range(i16, grid_width - 1)", "@range(i16, @as(i16,grid_width) - 1)")
    )
    s = s.replace(
        "  sys_mod.unblock_cmd_stream();",
        """  timestamp.get_timestamp(&ended);
  for(@range(u16,3)) |i| {total_timing[i]=started[i];total_timing[i+3]=ended[i];}
  sys_mod.unblock_cmd_stream();""",
    )
    s = s.replace(
        "  @export_symbol(compute);\n  @rpc(@get_data_task_id(sys_mod.LAUNCH));",
        """  @export_symbol(main);
  @initialize_queue(@get_input_queue(2),.{.color=recv_A_color});
  @initialize_queue(@get_output_queue(2),.{.color=send_A_color});
  @initialize_queue(@get_input_queue(3),.{.color=recv_B_color});
  @initialize_queue(@get_output_queue(3),.{.color=send_B_color});""",
    )
    s = s.replace(
        '@export_symbol(A_ptr, "A");', '@export_symbol(host_A_ptr, "A");'
    ).replace('@export_symbol(B_ptr, "B");', '@export_symbol(host_B_ptr, "B");')
    s += "\nvar host_A_ptr:[*]f32=&Matrix_1;var host_B_ptr:[*]f32=&Matrix_2;\n"
    return s + """
const timestamp=@import_module("<time>");
var started=@zeros([3]u16);var ended=@zeros([3]u16);var total_timing=@zeros([6]u16);
const cd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{M_per_pe*N_per_pe}->C[i]});
fn main() void {
 timestamp.enable_tsc();timestamp.get_timestamp(&started);
 A_ptr=&Matrix_1;B_ptr=&Matrix_2;Temp_ptr=&Matrix_3;
 A_full_dsd=@set_dsd_base_addr(A_full_dsd,A_ptr);B_full_dsd=@set_dsd_base_addr(B_full_dsd,B_ptr);
 B_row_dsd=@set_dsd_base_addr(B_row_dsd,B_ptr);Temp_full_dsd=@set_dsd_base_addr(Temp_full_dsd,Temp_ptr);
 C_row_dsd=@set_dsd_base_addr(C_row_dsd,&C);@fmovs(cd,0.0);compute();
}
var total_ptr:[*]u16=&total_timing;
comptime {@export_symbol(total_ptr,"total_timing");}
"""


def main(bundle):
    import numpy as np

    sys.path.insert(0, str(bundle / "implementation"))
    from integrity import verify_bundle
    from sdk_process import run_sdk
    from roundoff import check_matrix_roundoff

    verify_bundle(bundle)
    assert read(bundle / "qualification.json")["success"]
    upstream = ROOT / "projects/matrix_algorithms/upstream/Cannons_algorithm"
    root = (
        ROOT
        / "evidence"
        / (
            "cannon-native-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    indices = [0, 1]
    for name in (
        "schedule.json",
        "semantic.json",
        "runtime-options.json",
        "sdk-command.json",
    ):
        data = read(bundle / name)
        if isinstance(data, dict) and "epochs" in data:
            data["epochs"] = len(indices)
        (root / name).write_text(json.dumps(data) + "\n")
    b = [read(bundle / "batches.json")[i] for i in indices]
    (root / "batches.json").write_text(json.dumps(b) + "\n")
    shutil.copytree(
        bundle / "implementation",
        root / "implementation",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copyfile(upstream / "pe_program.csl", root / "original-pe.csl")
    shutil.copyfile(upstream / "layout.csl", root / "original-layout.csl")
    (root / "pe.csl").write_text(native_pe((upstream / "pe_program.csl").read_text()))
    layout = (bundle / "layout.csl").read_text()
    layout = re.sub(
        r'@export_name\("(?:history|witness|timing|progress|queue_last)",\[\*\](?:f32|u16),true\);',
        "",
        layout,
    )
    (root / "layout.csl").write_text(layout)
    shutil.copyfile(__file__, root / "driver.py")
    sif = Path(
        "/home/lingzhi/cerebras/sdk/2.10.1/sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif"
    )
    with sif.open("rb") as f:
        sdksha = hashlib.file_digest(f, "sha256").hexdigest()
    assert sdksha == "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"
    provenance = dict(
        source_bundle=str(bundle),
        source_results_sha256=sha(bundle / "results.json"),
        sdk_sha256=sdksha,
        selected_epochs=indices,
        scope="Pinned original Cannon compute and synchronous ring schedule; explicit SDK2.10.1 queue/parameter/launch migration, logical parity, stable exported input pointers, warm-reset and timestamp wrapper. HLS layout ABI and pure packing reused. No HLS histories/counters in native kernel.",
        files={
            str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()
        },
    )
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(root, flush=True)
    with (root / "sdk.log").open("w") as log:
        run_sdk(
            [
                "/home/lingzhi/cerebras/sdk/2.10.1/cs_python",
                str(root / "driver.py"),
                "--worker",
                str(root),
            ],
            root,
            dict(
                os.environ,
                SINGULARITYENV_CS_TARGET="SDR",
                SINGULARITYENV_PYTHONUNBUFFERED="1",
            ),
            log,
            600,
        )
    r = read(root / "results.json")
    original = read(bundle / "results.json")
    s = read(root / "schedule.json")
    m = read(root / "semantic.json")
    assert r["success"] and len(r["cases"]) == len(indices)
    rows = []
    for j, i in enumerate(indices):
        a, bb = [
            np.asarray(b[j][n["host"]], dtype=float).reshape(n["shape"])
            for n in m["nodes"][:2]
        ]
        got = np.asarray(r["cases"][j])
        hls = np.asarray(original["cases"][i][m["nodes"][3]["host"]]).reshape(got.shape)
        proof = check_matrix_roundoff(a, bb, got, a @ bb)
        t = np.asarray(r["timing"][j], np.int64)
        ticks = sum((t[:, :, k + 3] - t[:, :, k]) * (1 << (16 * k)) for k in range(3))
        assert np.all((ticks > 0) & (ticks < 2**32))
        ht = np.asarray(read(bundle / "audit.json")["total_cycles_per_epoch_pe"][i])
        rows.append(
            dict(
                epoch=i,
                native_roundoff=proof,
                outputs_exact_equal=bool(np.array_equal(got, hls)),
                hls_per_pe_cycles=ht.tolist(),
                native_per_pe_cycles=ticks.tolist(),
                hls_over_native_max_local_ratio=float(ht.max() / ticks.max()),
            )
        )
    (root / "comparison.json").write_text(
        json.dumps(
            dict(
                passed=True,
                comparisons=rows,
                scope=provenance["scope"]
                + " Maximum-local simulator intervals only; excludes H2D/D2H, not hardware or synchronized global latency.",
            ),
            indent=2,
        )
        + "\n"
    )
    print("PASS", [r["hls_over_native_max_local_ratio"] for r in rows], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    worker(a.bundle.resolve()) if a.worker else main(a.bundle.resolve())
