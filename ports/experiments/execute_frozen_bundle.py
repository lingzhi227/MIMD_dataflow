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
if (
    json.loads((root / "schedule.json").read_text())["profile"]
    == "mesh_cache_attention.v1"
):
    assert application_gate, "cache attention requires native stage gates"
    nodes = json.loads((root / "semantic.json").read_text())["nodes"]
    assert {o["node"] for o in application_gate.get("native_observations", [])} == {
        nodes[i]["id"] for i in (6, 7, 8, 9)
    }
    assert manifest.get("target_application_gate") == "target-application-gate.json"
    target = json.loads((root / "target-application-gate.json").read_text())
    assert target["passed"] and len(target["checks"]) == manifest["epochs"]
    assert all(
        c.get("all_stage_gates") and c.get("fixed_accuracy_passed")
        for gate in (application_gate, target)
        for c in gate["checks"]
    )
if (
    json.loads((root / "schedule.json").read_text())["profile"]
    == "mesh_projected_cache.v1"
):
    assert application_gate, "projected cache requires independent native stages"
    nodes = json.loads((root / "semantic.json").read_text())["nodes"]
    assert {o["node"] for o in application_gate.get("native_observations", [])} == {
        nodes[i]["id"] for i in (10, 11, 12, 14, 17, 18, 19, 20)
    }
    assert manifest.get("target_application_gate") == "target-application-gate.json"
    target = json.loads((root / "target-application-gate.json").read_text())
    assert target["passed"] and len(target["checks"]) == manifest["epochs"]
    assert all(
        c.get("all_stage_gates") and c.get("fixed_accuracy_passed")
        for gate in (application_gate, target)
        for c in gate["checks"]
    )
ffn_preflight = None
if (
    json.loads((root / "schedule.json").read_text())["profile"]
    == "mesh_batched_feed_forward.v1"
    and not application_gate
):
    gate_path = root / "ffn-preflight-gate.json"
    ffn_preflight = json.loads(gate_path.read_text())
    assert (
        ffn_preflight["passed"]
        and len(ffn_preflight["checks"])
        == len(ffn_preflight["target_checks"])
        == manifest["epochs"]
    )
    assert (
        ffn_preflight["manifest_sha256"]
        == hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    )
    assert (
        ffn_preflight["native_output_sha256"]
        == hashlib.sha256((root / "native-output.txt").read_bytes()).hexdigest()
    )
    assert len(ffn_preflight["native_observations"]) == 6
    nodes = json.loads((root / "semantic.json").read_text())["nodes"]
    roles = {
        nodes[i]["id"]: role
        for i, role in (
            (5, "normalized"),
            (6, "up"),
            (7, "gate"),
            (8, "activation"),
            (9, "hidden"),
            (10, "delta"),
        )
    }
    assert {o["node"] for o in ffn_preflight["native_observations"]} == set(roles)
    for observation in ffn_preflight["native_observations"]:
        location = root / ("native-observe-" + roles[observation["node"]])
        for name, digest in observation["files"].items():
            assert hashlib.sha256((location / name).read_bytes()).hexdigest() == digest
    ffn_preflight = dict(
        report=ffn_preflight, sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest()
    )
target_gate = None
if manifest.get("target_application_gate"):
    assert manifest["target_application_gate"] == "target-application-gate.json"
    target_gate = json.loads((root / "target-application-gate.json").read_text())
    assert target_gate["passed"] and len(target_gate["checks"]) == manifest["epochs"]
if (
    json.loads((root / "schedule.json").read_text())["profile"]
    == "mesh_batched_feed_forward.v1"
    and application_gate
):
    assert target_gate and len(application_gate.get("native_observations", [])) == 6
    nodes = json.loads((root / "semantic.json").read_text())["nodes"]
    assert {o["node"] for o in application_gate["native_observations"]} == {
        nodes[i]["id"] for i in (5, 6, 7, 8, 9, 10)
    }
    assert all(
        c.get("all_stage_gates") and c.get("fixed_accuracy_passed")
        for gate in (application_gate, target_gate)
        for c in gate["checks"]
    )
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
if (
    json.loads((root / "schedule.json").read_text())["profile"]
    == "mesh_input_attention_mixed.v1"
):
    assert application_gate and target_gate
    assert target_gate.get("kind") == "actual_prior_sdk_replay"
    nodes = json.loads((root / "semantic.json").read_text())["nodes"]
    assert {o["node"] for o in application_gate.get("native_observations", [])} == {
        nodes[i]["id"] for i in (11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 28)
    }
    assert "prior-sdk-replay-results.json" in manifest["files"]
    assert (
        target_gate["prior_results_sha256"]
        == hashlib.sha256(
            (root / "prior-sdk-replay-results.json").read_bytes()
        ).hexdigest()
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
    batched_ffn_preflight=ffn_preflight,
    independent_native_application_gate=application_gate,
    **{
        (
            "target_application_gate"
            if json.loads((root / "schedule.json").read_text())["profile"]
            == "mesh_input_attention_mixed.v1"
            else "predicted_target_application_gate"
        ): target_gate
    },
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
    progress_options = {}
    if "incremental_audit.py" in manifest.get("implementation", {}):
        from incremental_audit import make_checker

        checker = make_checker(root)
        if checker is not None:
            progress_options["progress_check"] = checker
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
            **progress_options,
        )
    report["audit"] = audit(root)
    if (root / "completed-audit.json").exists():
        report["completed_audit_sha256"] = hashlib.sha256(
            (root / "completed-audit.json").read_bytes()
        ).hexdigest()
    report["success"] = True
except BaseException as e:
    report["error"] = repr(e)
    raise
finally:
    (root / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
print("PASS", root, flush=True)
