"""Same-SDK primitive reduction baseline with diagnostic stores removed.

Reuses pinned SDK BLAS and the same SDK-collective wrapper. This compares HLS
observation overhead, not the unported CG application's original allreduce.
"""

import argparse, datetime, hashlib, json, os, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def worker(root):
    import subprocess, numpy as np

    sys.path.insert(0, str(root / "implementation"))
    from mesh_common import sdk_runtime
    from mesh_reduction_sdk import distribute
    from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder

    os.chdir(root)
    s = read(root / "schedule.json")
    m = read(root / "semantic.json")
    h, w, l = s["rows"], s["cols"], s["local_length"]
    subprocess.run(read(root / "sdk-command.json"), check=True)
    r = sdk_runtime(root)
    ids = {k: r.get_id(k) for k in ("x", "y", "result", "timing")}
    r.load()
    r.run()
    out = dict(success=False, cases=[], diagnostics=[], runtime_instances=1)
    for ep, b in enumerate(read(root / "batches.json")):
        for name, n in zip(("x", "y"), m["nodes"][:-2]):
            r.memcpy_h2d(
                ids[name],
                distribute(b[n["host"]], s).ravel(order="F"),
                0,
                0,
                w,
                h,
                l,
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.COL_MAJOR,
                nonblock=False,
            )
        r.launch("main", nonblock=False)
        d = {}
        for name, size, short in (("result", 1, False), ("timing", 6, True)):
            v = np.zeros(h * w * size, np.uint32 if short else np.float32)
            r.memcpy_d2h(
                v,
                ids[name],
                0,
                0,
                w,
                h,
                size,
                streaming=False,
                data_type=(
                    MemcpyDataType.MEMCPY_16BIT
                    if short
                    else MemcpyDataType.MEMCPY_32BIT
                ),
                order=MemcpyOrder.COL_MAJOR,
                nonblock=False,
            )
            d["replicas" if name == "result" else name] = v.reshape(
                h, w, size, order="F"
            ).tolist()
        out["cases"].append({m["nodes"][-1]["host"]: d["replicas"][0][0]})
        out["diagnostics"].append(d)
        (root / "results.json").write_text(json.dumps(out) + "\n")
        print("NATIVE REDUCTION EPOCH", ep + 1, "COMPLETE", flush=True)
    r.stop()
    out["success"] = True
    (root / "results.json").write_text(json.dumps(out) + "\n")


def main(bundle):
    import numpy as np

    sys.path.insert(0, str(ROOT / "toolchain"))
    from integrity import verify_bundle
    from sdk_process import run_sdk
    from mesh_reduction_sdk import check_result

    verify_bundle(bundle, implementation=False)
    s = read(bundle / "schedule.json")
    hls = read(bundle / "results.json")
    assert s["profile"] == "mesh_reduction.v1" and hls["success"]
    out = (
        ROOT
        / "evidence"
        / (
            "native-reduction-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    out.mkdir()
    for name in (
        "schedule.json",
        "semantic.json",
        "batches.json",
        "sdk-command.json",
        "runtime-options.json",
    ):
        if (bundle / name).exists():
            shutil.copy2(bundle / name, out / name)
    shutil.copytree(bundle / "implementation", out / "implementation")
    for name in ("layout.csl", "pe.csl", "scalar_allreduce.csl", "blas.csl"):
        code = (bundle / name).read_text()
        if name == "pe.csl":
            for var, ptr in (("witness", "wp"), ("progress", "pp")):
                code = re.sub(
                    r"\b" + var + r"\[[^\]]+\]\s*(?:\+=|=)\s*[^;]+;", "", code
                )
                code = re.sub(r"\bvar\s+" + var + r"\s*=\s*[^;]+;", "", code)
                code = re.sub(r"\bvar\s+" + ptr + r"\s*:[^;]+;", "", code)
                code = re.sub(
                    r"@export_symbol\(" + ptr + r',\s*"' + var + r'"\);', "", code
                )
                assert not re.search(r"\b" + var + r"\b", code), var
        if name == "layout.csl":
            for var in ("witness", "progress"):
                code = re.sub(r'@export_name\("' + var + r'",[^;]+;', "", code)
        (out / name).write_text(code)
    shutil.copy2(__file__, out / "driver.py")
    sif = Path(
        "/home/lingzhi/cerebras/sdk/2.10.1/sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif"
    )
    sdksha = sha(sif)
    assert sdksha == "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"
    manifest = dict(
        sdk_sha256=sdksha,
        comparison_bundle=str(bundle),
        comparison_results_sha256=sha(bundle / "results.json"),
        kind="Same SDK BLAS and row/column collective wrapper without HLS diagnostic stores. Not the original full CG reduction protocol.",
        files={str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out, flush=True)
    with (out / "sdk.log").open("w") as log:
        run_sdk(
            [
                "/home/lingzhi/cerebras/sdk/2.10.1/cs_python",
                str(out / "driver.py"),
                "--worker",
                str(out),
            ],
            out,
            dict(
                os.environ,
                SINGULARITYENV_CS_TARGET="SDR",
                SINGULARITYENV_PYTHONUNBUFFERED="1",
            ),
            log,
            600,
        )
    native = read(out / "results.json")
    m = read(out / "semantic.json")
    batches = read(out / "batches.json")
    assert native["success"] and len(native["cases"]) == s["epochs"]
    checks = []
    for b, c, d in zip(batches, native["cases"], native["diagnostics"]):
        v = c[m["nodes"][-1]["host"]]
        checks.append(
            check_result(s["operation"], [b[n["host"]] for n in m["nodes"][:-2]], v)
        )
        np.testing.assert_array_equal(
            np.asarray(d["replicas"]), np.full((s["rows"], s["cols"], 1), v[0])
        )

    def cycles(v):
        out = []
        for d in v["diagnostics"]:
            epoch = []
            for row in d["timing"]:
                for z in row:
                    delta = (
                        sum(int(z[i + 3]) << (16 * i) for i in range(3))
                        - sum(int(z[i]) << (16 * i) for i in range(3))
                    ) % (1 << 48)
                    assert 0 < delta < 1 << 32
                    epoch.append(delta)
            out.append(epoch)
        return out

    nt, ht = cycles(native), cycles(hls)
    result = dict(
        passed=True,
        epochs=s["epochs"],
        operation=s["operation"],
        numerical_checks=checks,
        max_local_ratios=[max(a) / max(b) for a, b in zip(ht, nt)],
        hls_per_pe_cycles=ht,
        native_per_pe_cycles=nt,
        outputs_exact_equal=hls["cases"] == native["cases"],
        native_results_sha256=sha(out / "results.json"),
        scope="Maximum local arithmetic plus collective interval, excludes H2D/D2H. Same SDK primitive adapter, diagnostic-store overhead only; not synchronized global latency or hardware performance.",
    )
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("passed", "max_local_ratios", "outputs_exact_equal")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    worker(a.bundle.resolve()) if a.worker else main(a.bundle.resolve())
