"""Independent native and predicted-target gates for the supplied-Q/K/V attention tail."""

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
from mesh_attention_tail import plan, inputs
from mesh_attention import reference as attention_reference
from mesh_feed_forward import normalized
from mesh_mlp import reference
from projection_reference import project
from attention_tail_fixtures import check


def seal_native(root):
    from native_observers import observe
    from application_gate import seal

    root = Path(root).resolve()
    m = json.loads((root / "semantic.json").read_text())
    s = plan(m)
    indices = dict(score=10, probability=11, attention=12, projection=13, delta=20)
    paths = [root / ("native-" + name + "-observation") for name in indices]
    values = [
        observe(root, m["nodes"][idx]["id"], path)[0]
        for idx, path in zip(indices.values(), paths)
    ]
    rows = iter(zip(*values))

    def checked(b, o):
        score, probability, attention, projection, delta = next(rows)
        return check(
            s["M"],
            s["N"],
            s["F"],
            s["epsilon"],
            s["scale"],
            b,
            o,
            attention,
            projection,
            delta,
            score=score,
            probability=probability,
        )

    return seal(
        root,
        ROOT / "tests/support/attention_tail_fixtures.py",
        checked,
        dict(M=s["M"], N=s["N"], F=s["F"]),
        native_observations=paths,
    )


def seal_target(root):
    root = Path(root)
    assert not (root / "target-application-gate.json").exists()
    m = json.loads((root / "semantic.json").read_text())
    s = plan(m)
    checks = []
    failure = None
    for epoch, b in enumerate(json.loads((root / "batches.json").read_text())):
        q, k, v, o, r, gamma, u, g, d = inputs(m, b)
        attention_result = attention_reference(s["attention_schedule"], q, k, v)
        score, probability, a = (
            attention_result[3],
            attention_result[6],
            attention_result[-1],
        )
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
                    s["scale"],
                    b,
                    {"output": final.ravel().tolist()},
                    a,
                    projection,
                    delta,
                    score=score,
                    probability=probability,
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


def device_checks(root):
    """Independent original-input review of actual SDK branch observations."""
    import numpy as np
    from mesh_common import unpack_tiles

    root = Path(root)
    s = json.loads((root / "schedule.json").read_text())
    bs = json.loads((root / "batches.json").read_text())
    r = json.loads((root / "results.json").read_text())
    assert r["success"] and len(bs) == len(r["cases"]) == len(r["diagnostics"])
    checks = []
    for b, o, d in zip(bs, r["cases"], r["diagnostics"]):

        def decoded(key, score=False):
            return unpack_tiles(
                np.asarray(d[key], np.uint16).view(np.float16),
                s["Mt"],
                s["Mt"] if score else s["Nt"],
                "F",
            )

        checks.append(
            check(
                s["M"],
                s["N"],
                s["F"],
                s["epsilon"],
                s["scale"],
                b,
                o,
                decoded("attention_snapshot"),
                decoded("projection_snapshot"),
                decoded("down_snapshot"),
                score=decoded("attention_logits", True),
                probability=decoded("attention_probability", True),
            )
        )
    return checks
