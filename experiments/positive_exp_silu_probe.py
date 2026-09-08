"""Eight-PE positive exp and original/stable half SiLU compound observations."""

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
            "positive-exp-silu-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    (root / "layout.csl").write_text(
        """const memcpy=@import_module("<memcpy/get_params>",.{.width=8,.height=1});
layout {@set_rectangle(8,1);for(@range(i16,8)) |x| {@set_tile_code(x,0,"pe.csl",.{.memcpy_params=memcpy.get_params(x)});}
@export_name("input",[*]f16,true);@export_name("value",[*]f16,true);@export_name("negative_silu",[*]f16,true);@export_name("stable_negative_silu",[*]f16,true);@export_name("positive_silu",[*]f16,true);@export_name("progress",[*]u16,true);@export_name("compute",fn()void);}
"""
    )
    (root / "pe.csl").write_text("""param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);const math=@import_module("<math>");
const length:i16=3968;
var input=@zeros([length]f16);var value=@zeros([length]f16);var negative_silu=@zeros([length]f16);var stable_negative_silu=@zeros([length]f16);var positive_silu=@zeros([length]f16);var progress=@zeros([1]u16);
fn compute() void {
 for(@range(i16,length)) |i| {
  const v=input[i];const e=math.exp_f16(v);const t=math.exp_f16(-v);
  value[i]=e;negative_silu[i]=(-v)/(@as(f16,1.0)+e);
  stable_negative_silu[i]=((-v)*t)/(@as(f16,1.0)+t);
  positive_silu[i]=v/(@as(f16,1.0)+t);
 }
 progress[0]+=1;sys.unblock_cmd_stream();
}
var ip:[*]f16=&input;var vp:[*]f16=&value;var np:[*]f16=&negative_silu;var sp:[*]f16=&stable_negative_silu;var ppv:[*]f16=&positive_silu;var pp:[*]u16=&progress;
comptime {@export_symbol(ip,"input");@export_symbol(vp,"value");@export_symbol(np,"negative_silu");@export_symbol(sp,"stable_negative_silu");@export_symbol(ppv,"positive_silu");@export_symbol(pp,"progress");@export_symbol(compute);}
""")
    bits = np.arange(0x7C00, dtype=np.uint16)
    values = bits.view(np.float16).astype(float)
    (root / "inputs.json").write_text(
        json.dumps([values.tolist(), values[::-1].tolist()]) + "\n"
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
    for src, dest in [
        (Path(__file__), "driver.py"),
        (ROOT / "experiments/probe_runtime.py", "probe_runtime.py"),
        (ROOT / "lib/Runtime/sdk_process.py", "sdk_process.py"),
    ]:
        shutil.copyfile(src, root / dest)
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                scope="All 31744 nonnegative finite half encodings including +0, 8 PEs, two changed warm calls. SDK positive exp, original half SiLU for both signs, and a nonpositive-exp alternative for negative SiLU. Raw half output bits include overflow/signed zero; no HLS application qualification.",
                sdk_math_source_hash=sha(ROOT / "third_party/references/sdk-math-2.10.1/math.csl"),
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
        k: runner.get_id(k)
        for k in (
            "input",
            "value",
            "negative_silu",
            "stable_negative_silu",
            "positive_silu",
            "progress",
        )
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
                3968,
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_16BIT,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False,
            )
            runner.launch("compute", nonblock=False)
            d = {}
            for name in ids:
                n = 1 if name == "progress" else 3968
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
            print("HALF EXP", epoch + 1, flush=True)
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
        execute(a.execute.resolve(), 1800)
    else:
        worker(a.worker.resolve())
