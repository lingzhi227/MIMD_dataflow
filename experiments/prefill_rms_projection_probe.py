"""Probe resident corrected RMSNorm followed by original Prefill Q projection.

This source experiment validates composition semantics; it is not an HLS port.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, difflib, json, os, re, shutil, subprocess, sys
from pathlib import Path
from probe_runtime import read, sha, verify, execute

ROOT = repository_root(__file__)


def prepare(hls, calls=6, isolate_exports=False):
    s = read(hls / "schedule.json")
    if s["profile"] == "mesh_normalized_matmul.v1":
        s = dict(s, profile="mesh_rms.v1", rows=s["P"], cols=s["P"])
    assert s["profile"] == "mesh_rms.v1" and s["rows"] == s["cols"]
    assert s["epsilon"] == 0.000001
    root = (
        ROOT
        / "validation/evidence"
        / (
            "prefill-rms-projection-"
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
    joined = original[:begin] + body + original[end:]
    first = joined.index("fn prefill_struct() void {")
    last = joined.index(
        "\n// --------------------------------------------------------------------------",
        first,
    )
    joined = (
        joined[:first] + "fn prefill_struct() void { hls_finish(); }\n" + joined[last:]
    )
    adapter = joined + """
var hls_norm=@zeros([seq_len_p_pe*dim_p_pe]f16);var hls_np:[*]f16=&hls_norm;
var hls_start=@zeros([3]u16);var hls_end=@zeros([3]u16);var hls_time=@zeros([6]u16);var hls_progress=@zeros([1]u16);
fn hls_rms_control() void {
 timestamp.enable_tsc();timestamp.get_timestamp(&hls_start);
 rmsnorm_x();
 for(@range(i16,seq_len_p_pe*dim_p_pe)) |i| {hls_norm[i]=X_norm_tile[i];}
 xq_matmul();
}
fn hls_finish() void {
 timestamp.get_timestamp(&hls_end);timestamp.disable_tsc();
 for(@range(i16,3)) |i| {hls_time[i]=hls_start[i];hls_time[i+3]=hls_end[i];}
 hls_progress[0]+=1;sys_mod.unblock_cmd_stream();
}
var hls_tp:[*]u16=&hls_time;var hls_pp:[*]u16=&hls_progress;
comptime {@export_symbol(ptr_XQ,"hls_result");@export_symbol(hls_np,"hls_normalized");@export_symbol(hls_tp,"hls_time");@export_symbol(hls_pp,"hls_progress");@export_symbol(hls_rms_control);}
"""
    if s["instrumentation"] == "counters":
        needle = (
            " for(@range(i16,seq_len_p_pe*dim_p_pe)) |i| {hls_norm[i]=X_norm_tile[i];}"
        )
        assert adapter.count(needle) == 1
        adapter = adapter.replace(needle, "")
        adapter = adapter.replace(
            "var hls_norm=@zeros([seq_len_p_pe*dim_p_pe]f16);",
            "var hls_norm=@zeros([1]f16);",
        )
    removed_exports = []
    if isolate_exports:
        removed_exports = [
            "K_weight",
            "V_weight",
            "freqs_sin",
            "freqs_cos",
            "O_weight",
            "UP_weight",
            "GATE_weight",
            "DOWN_weight",
            "time_memcpy",
            "time_ref",
        ]
        for name in removed_exports:
            pattern = r'@export_symbol\(\s*[^,;]+,\s*"' + name + r'"\s*\);'
            adapter, count = re.subn(pattern, "", adapter)
            assert count == 1, name
        assert adapter.count("@export_symbol(prefill_host);") == 1
        adapter = adapter.replace("@export_symbol(prefill_host);", "")
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
        + """ @export_name("hls_result",[*]f16,true);@export_name("hls_normalized",[*]f16,true);@export_name("hls_time",[*]u16,true);@export_name("hls_progress",[*]u16,true);@export_name("hls_rms_control",fn()void);
"""
        + layout[pos:]
    )
    if isolate_exports:
        for name in removed_exports + ["prefill_host"]:
            layout, count = re.subn(r'@export_name\("' + name + r'"[^;]+;', "", layout)
            assert count == 1, name
    (root / "layout.csl").write_text(layout)
    for name in ("semantic.json", "runtime-options.json"):
        shutil.copyfile(hls / name, root / name)
    (root / "schedule.json").write_text(json.dumps(s, indent=2) + "\n")
    import numpy as np

    inputs = read(hls / "batches.json")[:calls]
    rng = np.random.default_rng(210103)
    for epoch, b in enumerate(inputs):
        q = (
            np.eye(s["N"])
            if epoch in (0, 4)
            else rng.uniform(-1 / 16, 1 / 16, (s["N"], s["N"]))
        )
        b["q"] = np.asarray(q, np.float16).astype(float).ravel().tolist()
    (root / "inputs.json").write_text(json.dumps(inputs) + "\n")
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
                isolate_unused_exports=isolate_exports,
                removed_public_exports=removed_exports
                + (["prefill_host"] if isolate_exports else []),
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                source_files={
                    str(f.relative_to(src)): sha(f) for f in src.rglob("*.csl")
                },
                adaptation="Correct feature-indexed inverse to row vector. Weight host ownership corrected to PE columns. Original source communication and square/sum/math retained. Resident xq_matmul continuation; sampled mode copies normalized tile for observation before source in-place preshift; counter mode omits that copy. Stop after Q projection. Changed warm calls with freshly loaded X/W/Q; no intermediate host transfer. Not an HLS qualification or unmodified full inference.",
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
        for k in (
            "X",
            "W",
            "Q_weight",
            "hls_result",
            "hls_normalized",
            "hls_time",
            "hls_progress",
        )
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
            q = (
                np.asarray(b["q"], np.float16)
                .reshape(p, nt, p, nt)
                .transpose(0, 2, 1, 3)
            )
            cycle = list(range(0, p, 2)) + list(range(p - 1, 0, -2))
            pos = {v: i for i, v in enumerate(cycle)}
            q = np.asarray(
                [
                    [q[cycle[(pos[y] + pos[x]) % p], x] for x in range(p)]
                    for y in range(p)
                ]
            )
            for name, v, n in [
                ("X", packed, mt * nt),
                ("W", weights, nt),
                ("Q_weight", q, nt * nt),
            ]:
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
                ("hls_normalized", mt * nt if s["instrumentation"] == "sampled" else 1),
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
            print("RESIDENT RMS PROJECTION SOURCE", epoch + 1, flush=True)
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
    p.add_argument("--calls", type=int, choices=(2, 6), default=6)
    p.add_argument("--isolate-exports", action="store_true")
    a = p.parse_args()
    if a.prepare:
        prepare(a.prepare.resolve(), a.calls, a.isolate_exports)
    elif a.execute:
        execute(a.execute.resolve(), 1800)
    else:
        worker(a.worker.resolve())
