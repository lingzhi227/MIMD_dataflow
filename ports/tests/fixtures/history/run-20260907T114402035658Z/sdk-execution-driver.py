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
application_gate = None
if manifest.get("application_gate"):
    assert (
        manifest["application_gate"] == "application-gate.json"
        and "application-gate.json" in manifest["files"]
    )
    application_gate = json.loads((root / "application-gate.json").read_text())
    assert (
        application_gate["passed"]
        and len(application_gate["checks"]) == manifest["epochs"]
    ), "independent native application gate must pass before SDK"
    assert (
        application_gate["native_output_sha256"]
        == hashlib.sha256((root / "native-output.txt").read_bytes()).hexdigest()
    )
    assert (
        application_gate["reference_sha256"]
        == hashlib.sha256((root / "application-reference.py").read_bytes()).hexdigest()
    )
target_gate = None
if manifest.get("target_application_gate"):
    assert manifest["target_application_gate"] == "target-application-gate.json"
    target_gate = json.loads((root / "target-application-gate.json").read_text())
    assert target_gate["passed"] and len(target_gate["checks"]) == manifest["epochs"]
if (
    json.loads((root / "schedule.json").read_text())["profile"]
    == "mesh_feed_forward.v1"
):
    assert (
        application_gate and application_gate.get("native_observation") and target_gate
    )
    assert all(
        c.get("mlp_delta", {}).get("fixed_accuracy_passed")
        for c in application_gate["checks"]
    )
if (
    json.loads((root / "schedule.json").read_text())["profile"]
    == "mesh_prefill_tail.v1"
):
    assert application_gate and target_gate
    observations = application_gate.get("native_observations", [])
    nodes = json.loads((root / "semantic.json").read_text())["nodes"]
    assert {o["node"] for o in observations} == {nodes[7]["id"], nodes[14]["id"]}
    assert all(
        c.get(k, {}).get("fixed_accuracy_passed")
        for gate in (application_gate, target_gate)
        for c in gate["checks"]
        for k in ("projection", "mlp_delta")
    )
if (
    json.loads((root / "schedule.json").read_text())["profile"]
    == "mesh_attention_tail.v1"
):
    assert application_gate and target_gate
    observations = application_gate.get("native_observations", [])
    nodes = json.loads((root / "semantic.json").read_text())["nodes"]
    assert {o["node"] for o in observations} == {
        nodes[i]["id"] for i in (10, 11, 12, 13, 20)
    }
    assert all(
        c.get(k, {}).get("fixed_accuracy_passed")
        for gate in (application_gate, target_gate)
        for c in gate["checks"]
        for k in ("score", "probability", "attention", "projection", "mlp_delta")
    )
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
    independent_native_application_gate=application_gate,
    predicted_target_application_gate=target_gate,
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
