"""Matched RMS control: pinned Prefill with explicitly documented row-scale repair.

This is NOT an unmodified upstream correctness claim. Preserve full original
communication, local square/sum arithmetic, and patch only the proven index bug.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, difflib, json, os, shutil, subprocess, sys
from pathlib import Path
from probe_runtime import read, sha, verify, execute

ROOT = repository_root(__file__)


def prepare(hls):
    s = read(hls / "schedule.json")
    assert s["profile"] == "mesh_rms.v1" and s["rows"] == s["cols"]
    assert s["epsilon"] == 0.000001
    root = (
        ROOT
        / "validation/evidence"
        / (
            "rms-corrected-source-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    src = ROOT / "third_party/sources/waferllm/Prefill/src"
    shutil.copytree(src / "comm_lib", root / "comm_lib")
    original = (src / "prefill.csl").read_text()
    begin = original.index("fn rmsnorm_x() void {")
    end = original.index("\nfn xq_matmul()", begin)
    body = original[begin:end]
    needle = "    for (@range(i16, dim_p_pe)) |i| {\n        @fmulh(comp_dest_dsr_1, comp_src0_dsr_1, local_sum[i]);\n    }"
    assert body.count(needle) == 1
    body = body.replace(
        needle,
        "    @load_to_dsr(comp_src1_dsr_1, local_sum_dsd, .{ .save_address = false });\n    for (@range(i16, dim_p_pe)) |i| {\n        @fmulh(comp_dest_dsr_1, comp_src0_dsr_1, comp_src1_dsr_1);\n    }",
    )
    assert body.count("    prefill_struct();") == 1
    body = body.replace("    prefill_struct();", "")
    adapter = original[:begin] + body + original[end:] + """
var hls_start=@zeros([3]u16);var hls_end=@zeros([3]u16);var hls_time=@zeros([6]u16);var hls_progress=@zeros([1]u16);
fn hls_rms_control() void {
 timestamp.enable_tsc();timestamp.get_timestamp(&hls_start);
 rmsnorm_x();
 timestamp.get_timestamp(&hls_end);timestamp.disable_tsc();
 for(@range(i16,3)) |i| {hls_time[i]=hls_start[i];hls_time[i+3]=hls_end[i];}
 hls_progress[0]+=1;sys_mod.unblock_cmd_stream();
}
var hls_tp:[*]u16=&hls_time;var hls_pp:[*]u16=&hls_progress;
comptime {@export_symbol(ptr_X_norm,"hls_result");@export_symbol(hls_tp,"hls_time");@export_symbol(hls_pp,"hls_progress");@export_symbol(hls_rms_control);}
"""
    (root / "prefill.csl").write_text(adapter)
    (root / "source-adapter.diff").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                adapter.splitlines(True),
                fromfile="pinned/prefill.csl",
                tofile="corrected-control/prefill.csl",
            )
        )
    )
    layout = (src / "layout.csl").read_text()
    pos = layout.rfind("}")
    layout = (
        layout[:pos]
        + """ @export_name("hls_result",[*]f16,true);@export_name("hls_time",[*]u16,true);@export_name("hls_progress",[*]u16,true);@export_name("hls_rms_control",fn()void);
"""
        + layout[pos:]
    )
    (root / "layout.csl").write_text(layout)
    for name in ("schedule.json", "semantic.json", "runtime-options.json"):
        shutil.copyfile(hls / name, root / name)
    (root / "inputs.json").write_text(json.dumps(read(hls / "batches.json")[:2]) + "\n")
    p, mt, nt = s["cols"], s["Mt"], s["Nt"]
    cmd = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        f"--fabric-dims={p+7},{p+2}",
        "--fabric-offsets=4,1",
        f"--params=P:{p},dim_p_pe:{nt},pes_p_head:{p},pes_p_kv_head:{p},head_dim_p_pe:{nt},seq_len_p_pe:{mt},ffn_dim_p_pe:{nt}",
        "-o=out",
        "--memcpy",
        "--channels=1",
    ]
    (root / "sdk-command.json").write_text(json.dumps(cmd) + "\n")
    shutil.copyfile(__file__, root / "driver.py")
    shutil.copyfile(ROOT / "experiments/probe_runtime.py", root / "probe_runtime.py")
    shutil.copyfile(ROOT / "lib/Runtime/sdk_process.py", root / "sdk_process.py")
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                hls_bundle=str(hls.relative_to(ROOT)),
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                source_files={
                    str(f.relative_to(src)): sha(f) for f in src.rglob("*.csl")
                },
                adaptation="Correct feature-indexed inverse to row vector. Weight host ownership corrected to PE columns. Original source communication and square/sum/math retained. Added stage-only host entry, timestamps, progress and output export. Not unmodified upstream RMSNorm.",
                files={
                    str(f.relative_to(root)): sha(f)
                    for f in root.rglob("*")
                    if f.is_file()
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT), flush=True)


def worker(root):
    import numpy as np
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime,
        MemcpyDataType,
        MemcpyOrder,
        SimfabConfig,
        SdkTarget,
        get_platform,
    )
    from cerebras.sdk.sdk_utils import input_array_to_u32

    verify(root)
    os.chdir(root)
    s = read(root / "schedule.json")
    p, mt, nt = s["cols"], s["Mt"], s["Nt"]
    subprocess.run(read(root / "sdk-command.json"), check=True)
    runner = SdkRuntime(
        "out",
        get_platform(
            None, SimfabConfig(**read(root / "runtime-options.json")), SdkTarget.WSE3
        ),
    )
    ids = {
        k: runner.get_id(k)
        for k in ("X", "W", "hls_result", "hls_time", "hls_progress")
    }
    runner.load()
    runner.run()
    runner.launch("init_task", nonblock=False)
    result = dict(success=False, runtime_instances=1, cases=[])
    try:
        for epoch, b in enumerate(read(root / "inputs.json")):
            x = np.asarray(b["x"], np.float16).reshape(s["M"], s["N"])
            w = np.asarray(b["w"], np.float16)
            packed = x.reshape(p, mt, p, nt).transpose(0, 2, 3, 1)
            weights = np.repeat(w.reshape(1, p, nt), p, axis=0)
            for name, v, n in [("X", packed, mt * nt), ("W", weights, nt)]:
                runner.memcpy_h2d(
                    ids[name],
                    input_array_to_u32(v.ravel(), 1, 1),
                    0,
                    0,
                    p,
                    p,
                    n,
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_16BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
            runner.launch("hls_rms_control", nonblock=False)
            case = {}
            for name, n in [
                ("hls_result", mt * nt),
                ("hls_time", 6),
                ("hls_progress", 1),
            ]:
                v = np.zeros(p * p * n, np.uint32)
                runner.memcpy_d2h(
                    v,
                    ids[name],
                    0,
                    0,
                    p,
                    p,
                    n,
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_16BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
                case[name] = v.astype(np.uint16).reshape(p, p, n).tolist()
            np.testing.assert_array_equal(case["hls_progress"], epoch + 1)
            result["cases"].append(case)
            (root / "results.json").write_text(json.dumps(result) + "\n")
            print("CORRECTED SOURCE RMS", epoch + 1, flush=True)
    finally:
        runner.stop()
    result["success"] = True
    (root / "results.json").write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", type=Path)
    g.add_argument("--execute", type=Path)
    g.add_argument("--worker", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.prepare.resolve())
    elif a.execute:
        execute(a.execute.resolve(), 900)
    else:
        worker(a.worker.resolve())
