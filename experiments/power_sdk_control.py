"""Power vector-DSD normalization versus scalar-loop normalization; identical distributed schedule."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, hashlib, json, os, re, shutil, sys
from pathlib import Path

ROOT = repository_root(__file__)


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def worker(root):
    for name, digest in read(root / "control-manifest.json")["files"].items():
        assert sha(root / name) == digest
    sys.path.insert(0, str(root / "implementation"))
    from mesh_cg_sdk import run

    run(root)


def main(bundle):
    import numpy as np

    sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
    from integrity import verify_bundle
    from sdk_process import run_sdk
    from power_fixtures import check

    verify_bundle(bundle, implementation=False)
    original = read(bundle / "results.json")
    assert original["success"]
    indices = [0, 7]
    root = (
        ROOT
        / "validation/evidence"
        / (
            "power-scalar-control-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    for name in [
        "schedule.json",
        "semantic.json",
        "runtime-options.json",
        "host-abi.json",
    ]:
        data = read(bundle / name)
        if "epochs" in data:
            data["epochs"] = len(indices)
        (root / name).write_text(json.dumps(data) + "\n")
    for name in ["batches.json", "sparse-packing.json"]:
        (root / name).write_text(
            json.dumps([read(bundle / name)[i] for i in indices]) + "\n"
        )
    shutil.copytree(bundle / "implementation", root / "implementation")
    for p in bundle.glob("*.csl"):
        code = p.read_text()
        if p.name == "kernel.csl":
            original_scale = "@fmuls(cg_xd,cg_yd,inverse);"
            assert code.count(original_scale) == 1
            code = code.replace(
                original_scale,
                "for(@range(u16,local_vec_sz)) |i| {cg_solution[i]=y_local_buf[i]*inverse;}",
            )
        (root / p.name).write_text(code)
    shutil.copyfile(__file__, root / "driver.py")
    shutil.copyfile(ROOT / "tests/support/power_fixtures.py", root / "independent_oracle.py")
    sif = Path(
        "/home/lingzhi/cerebras/sdk/2.10.1/sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif"
    )
    with sif.open("rb") as f:
        sdksha = hashlib.file_digest(f, "sha256").hexdigest()
    assert sdksha == "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"
    manifest = dict(
        sdk_sha256=sdksha,
        source_bundle=str(bundle),
        source_results_sha256=sha(bundle / "results.json"),
        selected_epochs=indices,
        scope="Same generated resident power schedule, replacing only vector DSD normalization with scalar-loop normalization in the SDK power-method style. All communication, iteration, diagnostics and input sizes retained.",
        files={
            str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()
        },
    )
    (root / "control-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
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
    got = read(root / "results.json")
    assert got["success"] and len(got["cases"]) == len(indices)

    def timing(d):
        t = np.array(d["cg_timing"], dtype=np.int64)
        v = sum((t[:, :, i + 3] - t[:, :, i]) * (1 << (16 * i)) for i in range(3))
        assert np.all(v > 0)
        return v.tolist()

    rows = []
    for j, i in enumerate(indices):
        o = got["cases"][j]
        ref = original["cases"][i]
        assert o["reason"] == ref["reason"] and o["iterations"] == ref["iterations"]
        for field in ["vector", "norms"]:
            np.testing.assert_allclose(o[field], ref[field], rtol=3e-5, atol=3e-6)
        full = np.array(timing(original["diagnostics"][i]))
        control = np.array(timing(got["diagnostics"][j]))
        rows.append(
            dict(
                source_epoch=i,
                check=check(read(root / "batches.json")[j], o),
                hls_per_pe_cycles=full.tolist(),
                control_per_pe_cycles=control.tolist(),
                scalar_over_vector_max_local_ratio=float(control.max() / full.max()),
                outputs_exact_equal=(o == ref),
            )
        )
    (root / "comparison.json").write_text(
        json.dumps(
            dict(
                passed=True,
                comparisons=rows,
                scope=manifest["scope"]
                + " Max local simulator intervals, excludes H2D/D2H; no hardware/global-latency or original seven-point power-method parity claim.",
            ),
            indent=2,
        )
        + "\n"
    )
    print("PASS", [r["scalar_over_vector_max_local_ratio"] for r in rows], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    worker(a.bundle.resolve()) if a.worker else main(a.bundle.resolve())
