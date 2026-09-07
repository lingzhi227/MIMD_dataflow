"""Observe pinned distributed Prefill RMS stage without changing its arithmetic."""

import argparse
import datetime
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
from probe_runtime import read, sha, verify, execute

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    import numpy as np

    source = ROOT / "projects/waferllm/upstream/Prefill/src"
    root = (
        ROOT
        / "evidence"
        / (
            "prefill-rms-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    shutil.copytree(source / "comm_lib", root / "comm_lib")
    original = (source / "prefill.csl").read_text()
    begin = original.index("fn rmsnorm_x() void {")
    end = original.index("\nfn xq_matmul()", begin)
    body = original[begin:end]
    assert body.count("prefill_struct();") == 1
    body = body.replace(
        "prefill_struct();", "hls_progress[0]+=1; sys_mod.unblock_cmd_stream();"
    )
    needle = "comm_mod.mv_allreduce_add_x(ptr_local_sum);"
    assert body.count(needle) == 1
    body = body.replace(
        needle,
        needle
        + "\n    for (@range(i16,seq_len_p_pe)) |i| { hls_reduced[i]=local_sum[i]; }",
    )
    adapter = original[:begin] + body + original[end:] + """
var hls_reduced=@zeros([seq_len_p_pe]f16);
var hls_progress=@zeros([1]u16);
var hls_reduced_ptr:[*]f16=&hls_reduced;
var hls_progress_ptr:[*]u16=&hls_progress;
comptime {
 @export_symbol(ptr_X_norm,"hls_result");
 @export_symbol(ptr_local_sum,"hls_inverse");
 @export_symbol(hls_reduced_ptr,"hls_reduced");
 @export_symbol(hls_progress_ptr,"hls_progress");
 @export_symbol(rmsnorm_x);
}
"""
    (root / "prefill.csl").write_text(adapter)
    layout = (source / "layout.csl").read_text()
    pos = layout.rfind("}")
    layout = layout[:pos] + """ @export_name("hls_result",[*]f16,true);
 @export_name("hls_inverse",[*]f16,true);
 @export_name("hls_reduced",[*]f16,true);
 @export_name("hls_progress",[*]u16,true);
 @export_name("rmsnorm_x",fn()void);
""" + layout[pos:]
    (root / "layout.csl").write_text(layout)
    (root / "source-adapter.diff").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                adapter.splitlines(True),
                fromfile="pinned/prefill.csl",
                tofile="probe/prefill.csl",
            )
        )
    )
    p, t, n = 8, 8, 64
    rng = np.random.default_rng(2101)
    x = rng.uniform(-0.5, 0.5, (n, n)).astype(np.float16)
    scales = (1 + np.arange(n) % t).astype(np.float16) / np.float16(t)
    pattern = np.tile(np.asarray([-0.5, 0.25, 0.5, -0.25], np.float16), n // 4)
    row_scaled = (scales[:, None] * pattern[None, :]).astype(np.float16)
    (root / "inputs.json").write_text(
        json.dumps(
            [
                {"x": v.ravel().tolist(), "w": np.ones(n, np.float16).tolist()}
                for v in (x, row_scaled, np.zeros_like(x))
            ]
        )
        + "\n"
    )
    (root / "sdk-command.json").write_text(
        json.dumps(
            [
                "cslc",
                "layout.csl",
                "--arch=wse3",
                "--fabric-dims=15,10",
                "--fabric-offsets=4,1",
                "--params=P:8,dim_p_pe:8,pes_p_head:8,pes_p_kv_head:8,head_dim_p_pe:8,seq_len_p_pe:8,ffn_dim_p_pe:8",
                "-o=out",
                "--memcpy",
                "--channels=1",
            ]
        )
        + "\n"
    )
    shutil.copyfile(__file__, root / "driver.py")
    shutil.copyfile(ROOT / "experiments/probe_runtime.py", root / "probe_runtime.py")
    shutil.copyfile(ROOT / "toolchain/sdk_process.py", root / "sdk_process.py")
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                prepared_only=True,
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                source_files={
                    str(f.relative_to(source)): sha(f) for f in source.rglob("*.csl")
                },
                adaptation="Only selected RMS continuation becomes host completion; added raw reduced-sum observation and exports. Arithmetic and original communication modules unchanged. All-one weights isolate normalization from host weight-layout concerns.",
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
    subprocess.run(read(root / "sdk-command.json"), check=True)
    p, t, n = 8, 8, 64
    runner = SdkRuntime(
        "out",
        get_platform(
            None,
            SimfabConfig(suppress_trace=True, num_threads=8, dump_core=True),
            SdkTarget.WSE3,
        ),
    )
    ids = {
        name: runner.get_id(name)
        for name in (
            "X",
            "W",
            "hls_result",
            "hls_inverse",
            "hls_reduced",
            "hls_progress",
        )
    }
    runner.load()
    runner.run()
    runner.launch("init_task", nonblock=False)
    results = dict(success=False, runtime_instances=1, cases=[])
    try:
        for epoch, b in enumerate(read(root / "inputs.json")):
            x = np.asarray(b["x"], np.float16).reshape(n, n)
            packed = x.reshape(p, t, p, t).transpose(0, 2, 3, 1).reshape(p, p, t * t)
            # Exactly the original host layout; all-one W removes its axis ambiguity.
            weights = np.tile(np.asarray(b["w"], np.float16).reshape(p, t), reps=(1, p))
            for name, values, count in (("X", packed, t * t), ("W", weights, t)):
                runner.memcpy_h2d(
                    ids[name],
                    input_array_to_u32(values.ravel(), 1, 1),
                    0,
                    0,
                    p,
                    p,
                    count,
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_16BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
            runner.launch("rmsnorm_x", nonblock=False)
            result = {}
            for name, count in (
                ("hls_result", t * t),
                ("hls_inverse", t),
                ("hls_reduced", t),
                ("hls_progress", 1),
            ):
                raw = np.zeros(p * p * count, np.uint32)
                runner.memcpy_d2h(
                    raw,
                    ids[name],
                    0,
                    0,
                    p,
                    p,
                    count,
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_16BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
                result[name] = raw.astype(np.uint16).reshape(p, p, count).tolist()
            np.testing.assert_array_equal(np.asarray(result["hls_progress"]), epoch + 1)
            results["cases"].append(result)
            (root / "results.json").write_text(json.dumps(results) + "\n")
            print("RMS SOURCE", epoch + 1, flush=True)
    finally:
        runner.stop()
    results["success"] = True
    (root / "results.json").write_text(json.dumps(results) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--execute", type=Path)
    group.add_argument("--worker", type=Path)
    a = parser.parse_args()
    if a.prepare:
        prepare()
    elif a.execute:
        execute(a.execute.resolve())
    else:
        worker(a.worker.resolve())
