"""SDK2.10.1 half transport, fused/split rounding and DSR arithmetic probe."""

import datetime, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PE = r"""
param memcpy_params;param N:i16;
const sys_mod=@import_module("<memcpy/memcpy>",memcpy_params);
var a=@zeros([N]f16);var b=@zeros([N]f16);var c=@zeros([N]f16);
var fused=@zeros([N]f16);var split=@zeros([N]f16);var product=@zeros([N]f16);var dsr=@zeros([N]f16);
const ad=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->a[i]});
const cd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->c[i]});
const fd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->fused[i]});
const sd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->split[i]});
const pd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->product[i]});
const dd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{1}->dsr[i]});
const dest=@get_dsr(dsr_dest,1);const src0=@get_dsr(dsr_src0,1);const src1=@get_dsr(dsr_src1,1);
fn main() void {
 for(@range(i16,N)) |i| {
  const aa=@increment_dsd_offset(ad,i,f16);const cc=@increment_dsd_offset(cd,i,f16);
  const ff=@increment_dsd_offset(fd,i,f16);const ss=@increment_dsd_offset(sd,i,f16);
  const pp=@increment_dsd_offset(pd,i,f16);const dr=@increment_dsd_offset(dd,i,f16);
  @fmach(ff,cc,aa,b[i]);@fmulh(pp,aa,b[i]);@faddh(ss,pp,c[i]);
  @load_to_dsr(dest,dr,.{.save_address=false});
  @load_to_dsr(src0,cc,.{.save_address=false});
  @load_to_dsr(src1,aa,.{.save_address=false});
  @fmach(dest,src0,src1,b[i]);
 }
 sys_mod.unblock_cmd_stream();
}
var pa:[*]f16=&a;var pb:[*]f16=&b;var pc:[*]f16=&c;
var pf:[*]f16=&fused;var ps:[*]f16=&split;var pdsr:[*]f16=&dsr;
comptime {@export_symbol(pa,"a");@export_symbol(pb,"b");@export_symbol(pc,"c");
 @export_symbol(pf,"fused");@export_symbol(ps,"split");@export_symbol(pdsr,"dsr");@export_symbol(main);}
"""
LAYOUT = r"""
param N:u16;
const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});
layout {
 @set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.N=@as(i16,N),.memcpy_params=memcpy.get_params(0)});
 @export_name("a",[*]f16,true);@export_name("b",[*]f16,true);@export_name("c",[*]f16,true);
 @export_name("fused",[*]f16,true);@export_name("split",[*]f16,true);@export_name("dsr",[*]f16,true);
 @export_name("main",fn()void);
}
"""


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


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
    from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view

    for name, digest in read(root / "provenance.json")["files"].items():
        assert sha(root / name) == digest
    os.chdir(root)
    subprocess.run(read("sdk-command.json"), check=True)
    runner = SdkRuntime(
        "out",
        get_platform(
            None,
            SimfabConfig(suppress_trace=True, num_threads=4, dump_core=True),
            SdkTarget.WSE3,
        ),
    )
    ids = {k: runner.get_id(k) for k in ["a", "b", "c", "fused", "split", "dsr"]}
    runner.load()
    runner.run()
    out = []
    try:
        for batch in read("inputs.json"):
            n = len(batch["a"])
            for k in ["a", "b", "c"]:
                v = np.asarray(batch[k], np.float16)
                raw = input_array_to_u32(v, 1, 1)
                runner.memcpy_h2d(
                    ids[k],
                    raw,
                    0,
                    0,
                    1,
                    1,
                    n,
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_16BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
            runner.launch("main", nonblock=False)
            r = {}
            for k in ["a", "b", "c", "fused", "split", "dsr"]:
                raw = np.zeros(n, np.uint32)
                runner.memcpy_d2h(
                    raw,
                    ids[k],
                    0,
                    0,
                    1,
                    1,
                    n,
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_16BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
                view = memcpy_view(raw, np.dtype(np.float16))
                np.testing.assert_array_equal(
                    view.view(np.uint16), raw.astype(np.uint16)
                )
                r[k] = view.view(np.uint16).tolist()
            out.append(r)
    finally:
        runner.stop()
    (root / "results.json").write_text(json.dumps(out) + "\n")


def main():
    import numpy as np

    sys.path.insert(0, str(ROOT / "toolchain"))
    from sdk_process import run_sdk

    root = (
        ROOT
        / "evidence"
        / (
            "half-arithmetic-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    cases = [
        (1 + 2**-10, 1 + 2**-10, -(1 + 2**-10)),
        (1, 1, 2**-11),
        (1 + 2**-10, 1, 2**-11),
        (2**-14, 0.5, 0),
        (2**-24, 1, 0),
        (65504, 2, -65504),
        (-0.0, 1, -0.0),
        (-2, 0.5, 1),
        (2**-14, 2**-11, 0),
        (3 * 2**-14, 2**-11, 0),
        (65504, -1, 65504),
        (1 + 2**-10, 1 + 2**-10, -(1 + 2**-9)),
    ]
    rng = np.random.default_rng(20260906)
    cases += list(
        map(tuple, (rng.integers(-16, 17, size=(65 - len(cases), 3)) / 16).tolist())
    )
    a = np.asarray(cases, np.float16)
    inputs = [
        {k: a[:, i].astype(float).tolist() for i, k in enumerate(["a", "b", "c"])},
        {
            k: np.roll(a[:, i], 7).astype(float).tolist()
            for i, k in enumerate(["a", "b", "c"])
        },
    ]
    (root / "inputs.json").write_text(json.dumps(inputs) + "\n")
    (root / "pe.csl").write_text(PE)
    (root / "layout.csl").write_text(LAYOUT)
    shutil.copyfile(__file__, root / "driver.py")
    cmd = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        "--fabric-dims=8,3",
        "--fabric-offsets=4,1",
        "--params=N:65",
        "-o=out",
        "--memcpy",
        "--channels=1",
        "--width-west-buf=0",
        "--width-east-buf=0",
    ]
    (root / "sdk-command.json").write_text(json.dumps(cmd) + "\n")
    sif = Path(
        "/home/lingzhi/cerebras/sdk/2.10.1/sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif"
    )
    with sif.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    assert digest == "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                kind="target_semantics_probe_not_application_port",
                sdk_sha256=digest,
                files={p.name: sha(p) for p in root.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )
    print(root, flush=True)
    with (root / "sdk.log").open("w") as log:
        run_sdk(
            [
                "/home/lingzhi/cerebras/sdk/2.10.1/cs_python",
                str(root / "driver.py"),
                str(root),
            ],
            root,
            dict(
                os.environ,
                SINGULARITYENV_CS_TARGET="SDR",
                SINGULARITYENV_PYTHONUNBUFFERED="1",
            ),
            log,
            240,
        )
    results = read(root / "results.json")
    checks = []
    with np.errstate(over="ignore", invalid="ignore"):
        for b, r in zip(inputs, results):
            av, bv, cv = [
                np.asarray(b[k], np.float16).astype(float) for k in ["a", "b", "c"]
            ]
            for k in ["a", "b", "c"]:
                np.testing.assert_array_equal(
                    r[k], np.asarray(b[k], np.float16).view(np.uint16)
                )
            np.testing.assert_array_equal(r["fused"], r["dsr"])
            expected_fused = np.asarray(av * bv + cv, np.float16).view(np.uint16)
            expected_split = np.asarray(
                np.asarray(av * bv, np.float16).astype(float) + cv, np.float16
            ).view(np.uint16)
            checks.append(
                dict(
                    transport_exact=True,
                    dsr_matches_direct=True,
                    ieee_fused_exact=bool(np.array_equal(r["fused"], expected_fused)),
                    ieee_split_exact=bool(np.array_equal(r["split"], expected_split)),
                    fused_mismatches=[
                        dict(index=i, actual=int(v), expected=int(expected_fused[i]))
                        for i, v in enumerate(r["fused"])
                        if v != expected_fused[i]
                    ],
                    split_mismatches=[
                        dict(index=i, actual=int(v), expected=int(expected_split[i]))
                        for i, v in enumerate(r["split"])
                        if v != expected_split[i]
                    ],
                    fused_vs_split_different_indices=[
                        i
                        for i, (x, y) in enumerate(zip(r["fused"], r["split"]))
                        if x != y
                    ],
                )
            )
    (root / "observations.json").write_text(
        json.dumps(
            dict(
                completed=True,
                cases=checks,
                scope="65half entries including odd transport extent, normal/subnormal, ties, cancellation and overflow; two changed warm calls. Direct and DSR f16 FMA. Not full numerical-kernel qualification.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    worker(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else main()
