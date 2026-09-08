"""Independent native and predicted-target gates for the supplied-attention tail."""

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
from mesh_prefill_tail import plan, inputs
from mesh_feed_forward import normalized
from mesh_mlp import reference
from projection_reference import project
from prefill_tail_fixtures import check


def seal_native(root):
    from native_observers import observe
    from application_gate import seal

    root = Path(root).resolve()
    m = json.loads((root / "semantic.json").read_text())
    s = plan(m)
    pp = root / "native-projection-observation"
    dp = root / "native-delta-observation"
    projection, _ = observe(root, m["nodes"][7]["id"], pp)
    delta, _ = observe(root, m["nodes"][14]["id"], dp)
    pairs = iter(zip(projection, delta))

    def checked(b, o):
        pr, de = next(pairs)
        return check(s["M"], s["N"], s["F"], s["epsilon"], b, o, pr, de)

    return seal(
        root,
        ROOT / "tests/support/prefill_tail_fixtures.py",
        checked,
        dict(M=s["M"], N=s["N"], F=s["F"]),
        native_observations=[pp, dp],
    )


def seal_target(root):
    root = Path(root)
    assert not (root / "target-application-gate.json").exists()
    m = json.loads((root / "semantic.json").read_text())
    s = plan(m)
    checks = []
    failure = None
    for epoch, b in enumerate(json.loads((root / "batches.json").read_text())):
        a, o, r, gamma, u, g, d = inputs(m, b)
        projection, *_ = project(s["projection_prelude"], a, o)
        z = (projection + r).astype("float16").astype(float)
        delta, _ = reference(s, normalized(s, z, gamma), u, g, d)
        final = (z + delta).astype("float16").astype(float)
        try:
            checks.append(
                check(
                    s["M"],
                    s["N"],
                    s["F"],
                    s["epsilon"],
                    b,
                    {"output": final.ravel().tolist()},
                    projection,
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
        scope="Predicted target arithmetic; projection/delta/final independently checked against original-input fsum/sqrt/exp. Actual C++ observations are separately sealed; no SDK execution in this gate.",
    )
    (root / "target-application-gate.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    shutil.copyfile(__file__, root / "target-application-gate-driver.py")
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
                    failure=failure,
                    sdk_started=False,
                )
            )
            + "\n"
        )
    assert result["passed"], failure
    return result
