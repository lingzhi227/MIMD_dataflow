"""Execute original Prefill softmax score stage including all-negative stress."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, difflib, json, os, shutil, subprocess
from pathlib import Path
from probe_runtime import read, sha, verify, execute

ROOT = repository_root(__file__)


def prepare():
    import numpy as np

    root = (
        ROOT
        / "validation/evidence"
        / (
            "prefill-softmax-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    src = ROOT / "third_party/sources/waferllm/Prefill/src"
    shutil.copytree(src / "comm_lib", root / "comm_lib")
    original = (src / "prefill.csl").read_text()
    begin = original.index("fn softmax_score() void {")
    end = original.index("\nfn output_matmul()", begin)
    body = original[begin:end]
    assert body.count("    prefill_struct();") == 1
    body = body.replace(
        "    prefill_struct();", "    hls_progress[0]+=1;sys_mod.unblock_cmd_stream();"
    )
    needle = "    comm_mod.mv_allreduce_add_x(ptr_local_sum);"
    assert body.count(needle) == 1
    body = body.replace(
        needle,
        needle + "\n    for(@range(i16,seq_len_p_pe)) |i| {hls_sum[i]=local_sum[i];}",
    )
    adapter = original[:begin] + body + original[end:] + """
var hls_progress=@zeros([1]u16);var hls_sum=@zeros([seq_len_p_pe]f16);
var hls_pp:[*]u16=&hls_progress;var hls_sp:[*]f16=&hls_sum;
comptime {@export_symbol(ptr_score,"hls_score");@export_symbol(ptr_local_max,"hls_max");@export_symbol(ptr_local_sum,"hls_inverse");@export_symbol(ptr_seqLen_seqLen_tmp,"hls_exp");@export_symbol(hls_sp,"hls_sum");@export_symbol(hls_pp,"hls_progress");@export_symbol(softmax_score);}
"""
    (root / "prefill.csl").write_text(adapter)
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
    layout = (src / "layout.csl").read_text()
    pos = layout.rfind("}")
    layout = (
        layout[:pos]
        + """ @export_name("hls_score",[*]f16,true);@export_name("hls_max",[*]f16,true);@export_name("hls_inverse",[*]f16,true);@export_name("hls_exp",[*]f16,true);@export_name("hls_sum",[*]f16,true);@export_name("hls_progress",[*]u16,true);@export_name("softmax_score",fn()void);
"""
        + layout[pos:]
    )
    (root / "layout.csl").write_text(layout)
    rng = np.random.default_rng(210102)
    values = [
        rng.uniform(-2, 2, (64, 64)),
        np.full((64, 64), -1024.0),
        np.zeros((64, 64)),
    ]
    (root / "inputs.json").write_text(
        json.dumps(
            [np.asarray(v, np.float16).astype(float).ravel().tolist() for v in values]
        )
        + "\n"
    )
    cmd = [
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
    (root / "sdk-command.json").write_text(json.dumps(cmd) + "\n")
    for source, dest in [
        (Path(__file__), "driver.py"),
        (ROOT / "experiments/probe_runtime.py", "probe_runtime.py"),
        (ROOT / "lib/Runtime/sdk_process.py", "sdk_process.py"),
    ]:
        shutil.copyfile(source, root / dest)
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                source_files={
                    str(f.relative_to(src)): sha(f) for f in src.rglob("*.csl")
                },
                adaptation="Only stage completion/exports and reduced-sum observation; original arithmetic and communication unchanged. Random, all-negative-1024, then zero changed warm calls. Not an HLS qualification.",
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
    runner = SdkRuntime(
        "out",
        get_platform(
            None,
            SimfabConfig(suppress_trace=True, num_threads=8, dump_core=True),
            SdkTarget.WSE3,
        ),
    )
    ids = {
        k: runner.get_id(k)
        for k in (
            "hls_score",
            "hls_max",
            "hls_inverse",
            "hls_exp",
            "hls_sum",
            "hls_progress",
        )
    }
    runner.load()
    runner.run()
    runner.launch("init_task", nonblock=False)
    result = dict(success=False, runtime_instances=1, cases=[])
    try:
        for epoch, values in enumerate(read(root / "inputs.json")):
            x = np.asarray(values, np.float16).reshape(8, 8, 8, 8).transpose(0, 2, 3, 1)
            runner.memcpy_h2d(
                ids["hls_score"],
                input_array_to_u32(x.ravel(), 1, 1),
                0,
                0,
                8,
                8,
                64,
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_16BIT,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False,
            )
            runner.launch("softmax_score", nonblock=False)
            case = {}
            for name, n in [
                ("hls_score", 64),
                ("hls_max", 8),
                ("hls_inverse", 8),
                ("hls_exp", 64),
                ("hls_sum", 8),
                ("hls_progress", 1),
            ]:
                v = np.zeros(64 * n, np.uint32)
                runner.memcpy_d2h(
                    v,
                    ids[name],
                    0,
                    0,
                    8,
                    8,
                    n,
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_16BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
                case[name] = v.astype(np.uint16).reshape(8, 8, n).tolist()
            np.testing.assert_array_equal(case["hls_progress"], epoch + 1)
            result["cases"].append(case)
            (root / "results.json").write_text(json.dumps(result) + "\n")
            print("SOURCE SOFTMAX", epoch + 1, flush=True)
    finally:
        runner.stop()
    result["success"] = True
    (root / "results.json").write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--execute", type=Path)
    g.add_argument("--worker", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.execute:
        execute(a.execute.resolve(), 900)
    else:
        worker(a.worker.resolve())
