"""Seal four actual native cache-attention stages and a separate predicted target preflight."""

import hashlib, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain"), str(ROOT / "experiments")]
from native_observers import observe
from application_gate import seal
from cache_attention_fixtures import check
from mesh_cache_attention_sdk import values
from cache_attention_reference import reference

INDICES = dict(score=6, probability=7, context=8, delta=9)


def seal_native(root):
    root = Path(root).resolve()
    m = json.loads((root / "semantic.json").read_text())
    s = json.loads((root / "schedule.json").read_text())
    paths = {k: root / ("native-" + k + "-observation") for k in INDICES}
    observed = {
        k: observe(root, m["nodes"][i]["id"], paths[k])[0] for k, i in INDICES.items()
    }
    rows = iter([{k: v[e] for k, v in observed.items()} for e in range(m["epochs"])])
    return seal(
        root,
        ROOT / "cache_attention_fixtures.py",
        lambda b, o: check(s["B"], s["N"], s["S"], b, o, next(rows)),
        dict(B=s["B"], N=s["N"], S=s["S"]),
        native_observations=list(paths.values()),
    )


def seal_target(root):
    root = Path(root).resolve()
    assert not (root / "target-application-gate.json").exists()
    m = json.loads((root / "semantic.json").read_text())
    s = json.loads((root / "schedule.json").read_text())
    bs = json.loads((root / "batches.json").read_text())
    checks = []
    failure = None
    for epoch, b in enumerate(bs):
        try:
            _, stages = reference(s, values(m, b))
            out = {"result": stages.pop("result").ravel().tolist()}
            checks.append(check(s["B"], s["N"], s["S"], b, out, stages))
        except (AssertionError, ValueError) as error:
            failure = dict(epoch=epoch, error=repr(error))
            break
    report = dict(
        passed=failure is None,
        checks=checks,
        failure=failure,
        scope="Predicted target order versus original-input stdlib fsum/sqrt/exp. Actual CSL is independently audited; this is a preflight only.",
    )
    (root / "target-application-gate.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    shutil.copyfile(__file__, root / "target-application-gate-driver.py")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["target_application_gate"] = "target-application-gate.json"
    for name in ("target-application-gate.json", "target-application-gate-driver.py"):
        manifest["files"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    assert report["passed"], f"target preflight failed: {failure}; SDK must not run"
    return report


def device_checks(root):
    import numpy as np
    from mesh_cache_attention_sdk import unshard

    root = Path(root)
    s = json.loads((root / "schedule.json").read_text())
    bs = json.loads((root / "batches.json").read_text())
    r = json.loads((root / "results.json").read_text())
    assert r["success"] and len(r["cases"]) == len(r["diagnostics"]) == len(bs)
    reports = []
    for batch, out, d in zip(bs, r["cases"], r["diagnostics"]):
        observed = {
            k: unshard(
                s,
                np.asarray(d[k], np.uint16).view(np.float16).astype(float),
                axis,
                width,
            )
            for k, axis, width in (
                ("score", "y", s["S"]),
                ("probability", "y", s["S"]),
                ("context", "x", s["N"]),
                ("delta", "y", s["N"]),
            )
        }
        reports.append(check(s["B"], s["N"], s["S"], batch, out, observed))
    return reports
