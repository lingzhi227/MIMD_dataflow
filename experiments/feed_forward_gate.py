"""Reject predicted original-input FFN delta failures before spending an SDK run."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import hashlib, json, shutil, sys
from pathlib import Path

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from mesh_feed_forward import plan, inputs, normalized
from mesh_mlp import reference
from feed_forward_fixtures import check


def seal_native(root):
    from native_observers import observe
    from application_gate import seal

    root = Path(root).resolve()
    m = json.loads((root / "semantic.json").read_text())
    s = plan(m)
    # Structural FFN verifier fixes node 10 as the down projection.
    observed, _ = observe(root, m["nodes"][10]["id"], root / "native-observation")
    values = iter(observed)
    return seal(
        root,
        ROOT / "tests/support/feed_forward_fixtures.py",
        lambda b, o: check(s["M"], s["N"], s["F"], s["epsilon"], b, o, next(values)),
        dict(M=s["M"], N=s["N"], F=s["F"]),
        native_observation=root / "native-observation",
    )


def seal_target(root):
    root = Path(root)
    assert not (root / "target-application-gate.json").exists()
    m = json.loads((root / "semantic.json").read_text())
    s = plan(m)
    bs = json.loads((root / "batches.json").read_text())
    checks = []
    failure = None
    for epoch, b in enumerate(bs):
        z, gamma, u, g, d = inputs(m, b)
        delta, _ = reference(s, normalized(s, z, gamma), u, g, d)
        output = (z + delta).astype("float16").astype(float)
        try:
            checks.append(
                check(
                    s["M"],
                    s["N"],
                    s["F"],
                    s["epsilon"],
                    b,
                    {"output": output.ravel().tolist()},
                    delta,
                )
            )
        except AssertionError as e:
            failure = dict(epoch=epoch, error=str(e))
            break
    result = dict(
        passed=failure is None,
        checks=checks,
        failure=failure,
        scope="Predicted target arithmetic versus original-input fsum/sqrt/exp; separate delta gate. Not actual SDK observations; actual native delta has its own gate.",
    )
    (root / "target-application-gate.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    shutil.copyfile(Path(__file__), root / "target-application-gate-driver.py")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["target_application_gate"] = "target-application-gate.json"
    for name in ("target-application-gate.json", "target-application-gate-driver.py"):
        manifest["files"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if failure:
        (root / "stage.json").write_text(
            json.dumps(
                dict(
                    stage="target_application_failed",
                    epoch=failure["epoch"],
                    sdk_started=False,
                    native_application_passed=True,
                )
            )
            + "\n"
        )
    assert result[
        "passed"
    ], f"target application accuracy failed at {failure}; SDK must not run"
    return result
