"""Matched CSL scalar-loop versus @map SDK exp, with raw bits and local timing."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, os, shutil, subprocess
from pathlib import Path
from probe_runtime import read, sha, verify, execute

ROOT = repository_root(__file__)


def prepare():
    import numpy as np

    root = (
        ROOT
        / "validation/evidence"
        / (
            "exp-map-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    (root / "layout.csl").write_text(
        """const memcpy=@import_module("<memcpy/get_params>",.{.width=8,.height=1});
layout {@set_rectangle(8,1);for(@range(i16,8)) |x| {@set_tile_code(x,0,"pe.csl",.{.memcpy_params=memcpy.get_params(x)});}
@export_name("input",[*]f16,true);@export_name("scalar",[*]f16,true);@export_name("mapped",[*]f16,true);@export_name("timing",[*]u16,true);@export_name("progress",[*]u16,true);@export_name("compute",fn()void);}
"""
    )
    (root / "pe.csl").write_text(
        """param memcpy_params;const sys=@import_module("<memcpy/memcpy>",memcpy_params);const math=@import_module("<math>");const time=@import_module("<time>");
var input=@zeros([512]f16);var scalar=@zeros([512]f16);var mapped=@zeros([512]f16);var timing=@zeros([9]u16);var progress=@zeros([1]u16);
var begin=@zeros([3]u16);var middle=@zeros([3]u16);var end=@zeros([3]u16);
const id=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{512}->input[i]});const md=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{512}->mapped[i]});
fn exp_value(v:f16) f16 {return math.exp_f16(v);}
fn compute() void {
 time.enable_tsc();time.get_timestamp(&begin);
 for(@range(i16,512)) |i| {scalar[i]=exp_value(input[i]);}
 time.get_timestamp(&middle);@map(exp_value,id,md);time.get_timestamp(&end);time.disable_tsc();
 for(@range(i16,3)) |i| {timing[i]=begin[i];timing[3+i]=middle[i];timing[6+i]=end[i];}
 progress[0]+=1;sys.unblock_cmd_stream();
}
var ip:[*]f16=&input;var sp:[*]f16=&scalar;var mp:[*]f16=&mapped;var tp:[*]u16=&timing;var pp:[*]u16=&progress;
comptime {@export_symbol(ip,"input");@export_symbol(sp,"scalar");@export_symbol(mp,"mapped");@export_symbol(tp,"timing");@export_symbol(pp,"progress");@export_symbol(compute);}
"""
    )
    rng = np.random.default_rng(210103)
    x = rng.uniform(-16, 0, 4096).astype(np.float16)
    x[:8] = [-65504, -1024, -32, -17.328125, -1, -0.5, -0.0, 0.0]
    (root / "inputs.json").write_text(
        json.dumps(
            [
                x.astype(float).tolist(),
                x[::-1].astype(float).tolist(),
                np.zeros_like(x).astype(float).tolist(),
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
                "--fabric-dims=15,3",
                "--fabric-offsets=4,1",
                "-o=out",
                "--memcpy",
                "--channels=1",
            ]
        )
        + "\n"
    )
    for source, dest in [
        (Path(__file__), "driver.py"),
        (ROOT / "experiments/probe_runtime.py", "probe_runtime.py"),
        (ROOT / "lib/Runtime/sdk_process.py", "sdk_process.py"),
    ]:
        shutil.copyfile(source, root / dest)
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                scope="Same512half operands perPE,8PEs,3warmcalls. Explicitloop then@map using identical SDKmath helper; timestamps bracket each. Microbenchmark only, not application speedup.",
                sdk_math_source_sha256=sha(
                    ROOT / "third_party/references/sdk-math-2.10.1/math.csl"
                ),
                files={
                    str(f.relative_to(root)): sha(f)
                    for f in root.iterdir()
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
        k: runner.get_id(k) for k in ("input", "scalar", "mapped", "timing", "progress")
    }
    runner.load()
    runner.run()
    r = dict(success=False, runtime_instances=1, cases=[])
    try:
        for epoch, values in enumerate(read(root / "inputs.json")):
            x = np.asarray(values, np.float16)
            runner.memcpy_h2d(
                ids["input"],
                input_array_to_u32(x, 1, 1),
                0,
                0,
                8,
                1,
                512,
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_16BIT,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False,
            )
            runner.launch("compute", nonblock=False)
            d = {}
            for name, n in [
                ("input", 512),
                ("scalar", 512),
                ("mapped", 512),
                ("timing", 9),
                ("progress", 1),
            ]:
                v = np.zeros(8 * n, np.uint32)
                runner.memcpy_d2h(
                    v,
                    ids[name],
                    0,
                    0,
                    8,
                    1,
                    n,
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_16BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
                d[name] = v.astype(np.uint16).reshape(8, n).tolist()
            np.testing.assert_array_equal(
                np.asarray(d["input"], np.uint16).ravel(), x.view(np.uint16)
            )
            np.testing.assert_array_equal(d["progress"], epoch + 1)
            r["cases"].append(d)
            (root / "results.json").write_text(json.dumps(r) + "\n")
            print("EXP MAP", epoch + 1, flush=True)
    finally:
        runner.stop()
    r["success"] = True
    (root / "results.json").write_text(json.dumps(r) + "\n")


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
        execute(a.execute.resolve(), 1200)
    else:
        worker(a.worker.resolve())
