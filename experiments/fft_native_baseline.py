"""Matched original SDK FFT layout/RPC control; no HLS wrapper or phase counters."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = repository_root(__file__)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worker(root):
    import numpy as np

    sys.path.insert(0, str(root / "implementation"))
    from mesh_fft import inputs, pack, unpack, twiddles, interleave
    from mesh_fft_sdk import numerical_check
    from mesh_common import sdk_runtime
    from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder

    for name, digest in read(root / "provenance.json")["files"].items():
        assert sha(root / name) == digest
    os.chdir(root)
    s = read("schedule.json")
    m = read("semantic.json")
    p = s["rows"]
    n = s["N"]
    l = s["local_length"]
    subprocess.run(read("sdk-command.json"), check=True)
    r = sdk_runtime(root)
    ids = {k: r.get_id(k) for k in ("X", "twiddle_array", "fft_time")}
    r.load()
    r.run()

    def put(name, v, size):
        r.memcpy_h2d(
            ids[name],
            np.asarray(v, np.float32).ravel(),
            0,
            0,
            p,
            p,
            size,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )

    def get(name, size, short=False):
        v = np.zeros(p * p * size, np.uint32 if short else np.float32)
        r.memcpy_d2h(
            v,
            ids[name],
            0,
            0,
            p,
            p,
            size,
            streaming=False,
            data_type=(
                MemcpyDataType.MEMCPY_16BIT if short else MemcpyDataType.MEMCPY_32BIT
            ),
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )
        return v.reshape(p, p, size)

    put("twiddle_array", np.broadcast_to(twiddles(n), (p, p, n)), n)
    out = dict(
        success=False,
        cases=[],
        timing=[],
        checks=[],
        runtime_instances=1,
        packed_outputs=[],
    )
    for e, b in enumerate(read("batches.json")):
        put("X", pack(inputs(m, b), s), l)
        r.launch(
            "csfftExecC2C",
            np.int16(["backward", "ortho", "forward"].index(s["transform"]["norm"])),
            np.int16(s["transform"]["direction"] == "inverse"),
            nonblock=False,
        )
        packed = get("X", l)
        y = unpack(packed, s)
        out["packed_outputs"].append(packed.tolist())
        t = get("fft_time", 4, True).astype(np.uint64)
        out["cases"].append({s["output"]: interleave(y)})
        out["timing"].append(
            (t[:, :, 0] + (t[:, :, 1] << 16) + (t[:, :, 2] << 32)).tolist()
        )
        out["checks"].append(numerical_check(inputs(m, b), y, s["transform"]))
        Path("results.json").write_text(json.dumps(out) + "\n")
        print("NATIVE FFT", e + 1, flush=True)
    r.stop()
    out["success"] = True
    Path("results.json").write_text(json.dumps(out) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    if a.worker:
        return worker(a.bundle.resolve())
    b = a.bundle.resolve()
    q = read(b / "qualification.json")
    assert q["success"]
    sys.path.insert(0, str(b / "implementation"))
    from validate import audit

    fresh_audit = audit(b)
    assert fresh_audit["passed"]
    assert (
        read(b / "schedule.json").get("result_layout", "input_layout") == "input_layout"
    ), "Original SDK control restores the input layout"
    sif = Path(
        "/home/lingzhi/cerebras/sdk/2.10.1/sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif"
    )
    with sif.open("rb") as stream:
        sdk_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert (
        sdk_digest == q["sdk_sha256"]
    ), "Source control must use the qualified SDK image"
    root = (
        ROOT
        / "validation/evidence"
        / (
            "fft-native-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    shutil.copytree(b / "implementation", root / "implementation")
    for name in ("semantic.json", "schedule.json", "runtime-options.json"):
        shutil.copyfile(b / name, root / name)
    (root / "batches.json").write_text(json.dumps(read(b / "batches.json")[:2]) + "\n")
    (root / "layout.csl").write_text("""param N:u16;
param P:i16;
const memcpy=@import_module("<memcpy/get_params>",.{.width=P,.height=P});
const helper=@import_module("<kernels/fft/fft3d_layout>",.{.width=P,.memcpy=memcpy});
layout {@set_rectangle(P,P);helper.FFT_kernel(@as(u16,P),N,f32);}
""")
    s = read(root / "schedule.json")
    w = s["rows"]
    cmd = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        f"--fabric-dims={w+7},{w+2}",
        "--fabric-offsets=4,1",
        f"--params=N:{s['N']},P:{w}",
        "-o=out",
        "--memcpy",
        "--channels=1",
    ]
    (root / "sdk-command.json").write_text(json.dumps(cmd) + "\n")
    shutil.copyfile(__file__, root / "driver.py")
    files = [f for f in root.rglob("*") if f.is_file()]
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                hls_bundle=str(b),
                sdk_sha256=q["sdk_sha256"],
                scope="Unchanged actual SDK fft3d_layout/fft3d_rpc and internal FFT/transpose modules. Same HLS input packing and simulator settings; no HLS entry/callback counters or queue witnesses. Simulator local intervals only.",
                files={str(f.relative_to(root)): sha(f) for f in files},
            ),
            indent=2,
        )
        + "\n"
    )
    sys.path.insert(0, str(root / "implementation"))
    from sdk_process import run_sdk

    with (root / "sdk.log").open("w") as log:
        run_sdk(
            [
                "/home/lingzhi/cerebras/sdk/2.10.1/cs_python",
                str(root / "driver.py"),
                str(root),
                "--worker",
            ],
            root,
            dict(
                os.environ,
                SINGULARITYENV_CS_TARGET="SDR",
                SINGULARITYENV_PYTHONUNBUFFERED="1",
            ),
            log,
            1800,
        )
    import numpy as np

    native = read(root / "results.json")
    hls = read(b / "results.json")
    assert native["success"]
    comparisons = []
    for e, case in enumerate(native["cases"]):
        np.testing.assert_array_equal(
            np.asarray(native["packed_outputs"][e], np.float32).view(np.uint32),
            np.asarray(hls["diagnostics"][e]["packed_output"], np.float32).view(
                np.uint32
            ),
        )
        np.testing.assert_array_equal(
            np.asarray(case[s["output"]], np.float32).view(np.uint32),
            np.asarray(hls["cases"][e][s["output"]], np.float32).view(np.uint32),
        )
        nc = max(max(row) for row in native["timing"][e])
        hc = fresh_audit["cases"][e]["max_local_cycles"]
        comparisons.append(
            dict(
                epoch=e,
                output_bits_exact=True,
                packed_device_bits_exact=True,
                native_max_local_cycles=nc,
                hls_max_local_cycles=hc,
                hls_over_native_max_local=hc / nc,
            )
        )
    (root / "comparison.json").write_text(
        json.dumps(
            dict(
                passed=True,
                comparisons=comparisons,
                scope=read(root / "provenance.json")["scope"],
            ),
            indent=2,
        )
        + "\n"
    )
    print(root, flush=True)


if __name__ == "__main__":
    main()
