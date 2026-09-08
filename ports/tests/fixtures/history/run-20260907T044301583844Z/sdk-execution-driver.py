"""Execute a self-contained native-checked bundle without changing active toolchains."""

import argparse, datetime, hashlib, json, os, sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("bundle", type=Path)
parser.add_argument("--timeout", type=int, default=600)
args = parser.parse_args()
assert args.timeout > 0, "positive SDK timeout required"
root = args.bundle.resolve()
sys.path.insert(0, str(root / "implementation"))
from integrity import verify_bundle, verify_codegen
from sdk_process import run_sdk
from validate import audit

manifest = verify_bundle(root)
preflight = verify_codegen(root)
assert (
    json.loads((root / "stage.json").read_text())["stage"]
    == "native_passed_sdk_pending"
)
assert not (root / "sdk.log").exists(), "fresh execution required"
driver_copy = root / "sdk-execution-driver.py"
if driver_copy.resolve() != Path(__file__).resolve():
    assert not driver_copy.exists(), "fresh driver snapshot required"
    driver_copy.write_bytes(Path(__file__).read_bytes())
report = dict(
    success=False,
    frozen_codegen_preflight=preflight,
    timeout_seconds=args.timeout,
    bundle=str(root),
    source_sha256=manifest["source_sha256"],
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    native_build_location="preserved source bundle; executable and native/IR reference hashed in manifest",
    started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
)
try:
    sif = Path(
        "/home/lingzhi/cerebras/sdk/2.10.1/sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif"
    )
    with sif.open("rb") as stream:
        report["sdk_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    assert (
        report["sdk_sha256"]
        == "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"
    )
    with (root / "sdk.log").open("w") as log:
        run_sdk(
            [
                "/home/lingzhi/cerebras/sdk/2.10.1/cs_python",
                str(root / "implementation/sdk.py"),
                str(root),
            ],
            root,
            dict(
                os.environ,
                SINGULARITYENV_CS_TARGET="SDR",
                SINGULARITYENV_PYTHONUNBUFFERED="1",
            ),
            log,
            args.timeout,
        )
    report["audit"] = audit(root)
    report["success"] = True
except BaseException as e:
    report["error"] = repr(e)
    raise
finally:
    (root / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
print("PASS", root, flush=True)
