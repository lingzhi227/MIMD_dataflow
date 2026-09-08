"""Run the normal frozen compiler bundle with a measured, explicit SDK budget."""

import argparse, hashlib, json, os, shutil, sys, time, traceback
from pathlib import Path


def run(root, timeout):
    root = root.resolve()
    sys.path.insert(0, str(root / "implementation"))
    from integrity import verify_bundle, verify_codegen
    from sdk_process import run_sdk
    from incremental_audit import make_checker
    from mesh_projected_cache_ffn_sdk import audit

    verify_bundle(root)
    verify_codegen(root)
    assert not (root / "sdk.log").exists() and timeout >= 1
    controller = root / "execution-driver.py"
    assert not controller.exists()
    shutil.copy2(__file__, controller)
    image = Path(
        "/home/lingzhi/cerebras/sdk/2.10.1/sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif"
    )
    with image.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    assert digest == "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"
    cmd = [
        "/home/lingzhi/cerebras/sdk/2.10.1/cs_python",
        str(root / "implementation/sdk.py"),
        str(root),
    ]
    report = dict(
        success=False,
        timeout_seconds=timeout,
        sdk_sha256=digest,
        command=cmd,
        controller_sha256=hashlib.sha256(controller.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
        scope="Standard frozen HLS compiler SDK entry point; actual complete calls audited incrementally. Qualification requires independent source/mathematical/performance review.",
    )
    started = time.monotonic()
    try:
        with (root / "sdk.log").open("w") as log:
            run_sdk(
                cmd,
                root,
                os.environ.copy(),
                log,
                timeout,
                progress_check=make_checker(root),
            )
        result = audit(root)
        (root / "device-validation.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        report.update(
            success=True,
            results_sha256=hashlib.sha256(
                (root / "results.json").read_bytes()
            ).hexdigest(),
        )
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (root / "execution.json").write_text(json.dumps(report, indent=2) + "\n")
    print("STANDARD HLS SDK FULL PASS", root, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--timeout", type=int, default=21600)
    a = p.parse_args()
    run(a.root, a.timeout)
