"""Pinned WaferLLM compute/communication with only warm-entry and timing ABI adapters."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path

ROOT = repository_root(__file__)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worker(root):
    import numpy as np

    sys.path.insert(0, str(root / "implementation"))
    from mesh_common import sdk_runtime, unpack_tiles
    from mesh_twohop import inputs, pack
    from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder, MemcpyDataType
    from cerebras.sdk.sdk_utils import input_array_to_u32

    for name, digest in read(root / "provenance.json")["files"].items():
        assert sha(root / name) == digest
    os.chdir(root)
    subprocess.run(read("sdk-command.json"), check=True)
    s = read("schedule.json")
    m = read("semantic.json")
    p = s["P"]
    mt = s["Mt"]
    nt = s["Nt"]
    runner = sdk_runtime(root)
    ids = {k: runner.get_id(k) for k in ["X", "W", "res", "hls_total_timing"]}
    runner.load()
    runner.run()
    out = dict(success=False, cases=[], timing=[], runtime_instances=1)

    def get(name, n):
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
        return v.astype(np.uint16).reshape(p, p, n)

    try:
        for batch in read("batches.json"):
            for name, v in zip(["X", "W"], pack(*inputs(m, batch), p)):
                raw = input_array_to_u32(np.asarray(v, np.float16).ravel(), 1, 1)
                runner.memcpy_h2d(
                    ids[name],
                    raw,
                    0,
                    0,
                    p,
                    p,
                    v.shape[-1],
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_16BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
            runner.launch("hls_main", nonblock=False)
            out["cases"].append(
                unpack_tiles(get("res", mt * nt).view(np.float16), mt, nt, "F")
                .astype(float)
                .tolist()
            )
            out["timing"].append(get("hls_total_timing", 6).tolist())
            (root / "results.json").write_text(json.dumps(out) + "\n")
    finally:
        runner.stop()
    out["success"] = True
    (root / "results.json").write_text(json.dumps(out) + "\n")


def native_pe(source):
    s = "param logical_x:i16;param logical_y:i16;\n" + source.replace(
        "comm_lib/comm_pe.csl", "twohop_comm.csl"
    )
    for name in ["init_task", "meshgemm_entry", "meshgemm_host"]:
        s = s.replace("@export_symbol(" + name + ");", "")
    marker = "    f_memcpy_timestamps();"
    assert s.count(marker) == 1
    s = s.replace(
        marker,
        marker
        + "\n    for(@range(i16,3)) |i| {hls_total_timing[i]=tscStartBuffer[i];hls_total_timing[i+3]=tscEndBuffer[i];}",
    )
    return s + """
var hls_total_timing=@zeros([6]u16);
var ptt:[*]u16=&hls_total_timing;
fn hls_main() void {
 timestamp.enable_tsc();px=logical_x;py=logical_y;
 if(py%2==0){x_shift_step=py/2;x_shift_reverse=true;}
 else{x_shift_step=(py+1)/2;x_shift_reverse=false;}
 @block(x_finish_id);@block(y_finish_id);@block(jumpcast_finish_id);@block(next_step_id);
 ptr_out=&res_tile;out_dsd=@set_dsd_base_addr(out_dsd,ptr_out);
 total_repeat_times=1;total_warmup_times=0;repeat_times=0;step=0;
 meshgemm_entry();
}
comptime {@export_symbol(hls_main);@export_symbol(ptt,"hls_total_timing");}
"""


def main(bundle):
    import numpy as np

    sys.path.insert(0, str(bundle / "implementation"))
    from integrity import verify_bundle
    from sdk_process import run_sdk
    from binary16 import roundoff

    verify_bundle(bundle)
    assert read(bundle / "qualification.json")["success"]
    root = (
        ROOT
        / "validation/evidence"
        / (
            "twohop-native-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    upstream = ROOT / "third_party/sources/waferllm/MeshGEMM/src"
    indices = [0, 1]
    for name in [
        "schedule.json",
        "semantic.json",
        "runtime-options.json",
        "sdk-command.json",
    ]:
        data = read(bundle / name)
        if isinstance(data, dict) and "epochs" in data:
            data["epochs"] = len(indices)
        (root / name).write_text(json.dumps(data) + "\n")
    batches = [read(bundle / "batches.json")[i] for i in indices]
    (root / "batches.json").write_text(json.dumps(batches) + "\n")
    shutil.copytree(
        bundle / "implementation",
        root / "implementation",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(upstream, root / "original")
    (root / "pe.csl").write_text(native_pe((upstream / "meshgemm.csl").read_text()))
    for source, dest in [
        ("comm_lib/comm_pe.csl", "twohop_comm.csl"),
        ("comm_lib/comm_layout.csl", "twohop_routes.csl"),
    ]:
        shutil.copyfile(upstream / source, root / dest)
    layout = (bundle / "layout.csl").read_text()
    layout = re.sub(
        r'@export_name\("hls_(?:history|witness|timing|progress|queue)",\[\*\](?:f16|u16),true\);',
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
        scope="Pinned WaferLLM fd1c2daae37cd68706c03fc8009887ecee9900f8 original DSR FMA and two-hop communication unchanged; logical-coordinate warm-entry and timestamp ABI adapter, HLS layout ABI and pure packing reused. No HLS histories/witnesses/counters. Maximum-local simulator intervals, excludes host I/O; not hardware or synchronized global latency.",
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
    m = read(root / "semantic.json")
    assert r["success"] and len(r["cases"]) == len(indices)
    rows = []
    for j, i in enumerate(indices):
        a, b = [
            np.asarray(batches[j][n["host"]], dtype=float).reshape(n["shape"])
            for n in m["nodes"][:2]
        ]
        got = np.asarray(r["cases"][j])
        hls = np.asarray(original["cases"][i][m["nodes"][3]["host"]]).reshape(got.shape)
        np.testing.assert_array_equal(
            got.astype(np.float16).view(np.uint16),
            hls.astype(np.float16).view(np.uint16),
        )
        t = np.asarray(r["timing"][j], np.int64)
        ht = np.asarray(original["diagnostics"][i]["total_timing"], np.int64)
        ticks = sum((t[:, :, k + 3] - t[:, :, k]) * (1 << (16 * k)) for k in range(3))
        hcycles = sum(
            (ht[:, :, k + 3] - ht[:, :, k]) * (1 << (16 * k)) for k in range(3)
        )
        assert np.all((ticks > 0) & (ticks < 2**32))
        rows.append(
            dict(
                epoch=i,
                native_roundoff=roundoff(a, b, got),
                output_bits_exact=True,
                native_per_pe_cycles=ticks.tolist(),
                hls_per_pe_cycles=hcycles.tolist(),
                hls_over_native_max_local_ratio=float(hcycles.max() / ticks.max()),
            )
        )
    (root / "comparison.json").write_text(
        json.dumps(
            dict(passed=True, comparisons=rows, scope=provenance["scope"]), indent=2
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
