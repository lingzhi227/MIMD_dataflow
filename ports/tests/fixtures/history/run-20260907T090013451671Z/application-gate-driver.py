"""Seal actual C++ output versus an external original-input application checker."""

import hashlib, json, shutil
from pathlib import Path
from native_transport import parse_outputs


def seal(root, reference, check_output, dimensions, *, native_observation=None):
    root = Path(root)
    assert not (root / "application-gate.json").exists()
    batches = json.loads((root / "batches.json").read_text())
    native = parse_outputs((root / "native-output.txt").read_text())
    assert len(batches) == len(native)
    checks = []
    failure = None
    for epoch, (batch, output) in enumerate(zip(batches, native)):
        try:
            checks.append(check_output(batch, output))
        except AssertionError as error:
            failure = dict(epoch=epoch, error=repr(error))
            break
    shutil.copyfile(reference, root / "application-reference.py")
    shutil.copyfile(Path(__file__), root / "application-gate-driver.py")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    gate = dict(
        passed=failure is None,
        dimensions=dimensions,
        checks=checks,
        failure=failure,
        native_output_sha256=sha(root / "native-output.txt"),
        reference_sha256=sha(root / "application-reference.py"),
        gate_driver_sha256=sha(root / "application-gate-driver.py"),
        scope="Actual C++ stdout independently checked against original-input application mathematics before SDK",
    )
    observed_files = []
    if native_observation is not None:
        directory = Path(native_observation).resolve()
        directory.relative_to(root.resolve())
        observation = json.loads((directory / "observation.json").read_text())
        assert observation["passed"] and observation["source_sha256"] == sha(
            root / "source.cpp"
        )
        for name, digest in observation["files"].items():
            assert sha(directory / name) == digest
        observed_files = [p for p in directory.iterdir() if p.is_file()]
        gate["native_observation"] = dict(
            path=str(directory.relative_to(root.resolve())),
            report_sha256=sha(directory / "observation.json"),
            node=observation["node"],
            host=observation["host"],
        )
    (root / "application-gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["application_gate"] = "application-gate.json"
    for name in (
        "application-reference.py",
        "application-gate-driver.py",
        "application-gate.json",
    ):
        manifest["files"][name] = sha(root / name)
    for path in observed_files:
        manifest["files"][str(path.relative_to(root.resolve()))] = sha(path)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if failure:
        (root / "stage.json").write_text(
            json.dumps(
                dict(
                    stage="native_application_failed",
                    epoch=failure["epoch"],
                    compiler_native_equivalence_passed=True,
                    sdk_started=False,
                    application_gate="application-gate.json",
                )
            )
            + "\n"
        )
    assert gate[
        "passed"
    ], f"native application accuracy failed at epoch {failure['epoch']}; SDK must not run"
    return gate
