"""Seal actual native stage observations and independent full-graph target preflight."""

import hashlib, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain"), str(ROOT / "experiments")]
from native_observers import observe
from application_gate import seal
from projected_cache_fixtures import check
from mesh_projected_cache import values
from projected_cache_reference import reference

INDICES = dict(
    normalized=10,
    query=11,
    key_projection=12,
    rotated_query=14,
    score=17,
    probability=18,
    context=19,
    delta=20,
)


def seal_native(root):
    root = Path(root).resolve()
    s = json.loads((root / "schedule.json").read_text())
    m = json.loads((root / "semantic.json").read_text())
    paths = {k: root / ("native-" + k + "-observation") for k in INDICES}
    observed = {
        k: observe(root, m["nodes"][i]["id"], paths[k])[0] for k, i in INDICES.items()
    }
    rows = iter([{k: v[e] for k, v in observed.items()} for e in range(m["epochs"])])
    return seal(
        root,
        ROOT / "projected_cache_fixtures.py",
        lambda b, o: check(s["B"], s["N"], s["S"], b, o, next(rows)),
        dict(B=s["B"], N=s["N"], S=s["S"]),
        native_observations=list(paths.values()),
    )


def seal_target(root):
    root = Path(root).resolve()
    s, m, bs = [
        json.loads((root / n).read_text())
        for n in ("schedule.json", "semantic.json", "batches.json")
    ]
    assert not (root / "target-application-gate.json").exists()
    checks = []
    failure = None
    for epoch, b in enumerate(bs):
        try:
            _, stages = reference(s, values(m, b))
            stages = {k: v.ravel().tolist() for k, v in stages.items()}
            outputs = dict(
                result=stages.pop("result"),
                new_key=stages.pop("rotated_key"),
                new_value=stages.pop("value_projection"),
            )
            checks.append(check(s["B"], s["N"], s["S"], b, outputs, stages))
        except (AssertionError, ValueError) as error:
            failure = dict(epoch=epoch, error=repr(error))
            break
    report = dict(
        passed=failure is None,
        checks=checks,
        failure=failure,
        scope="Predicted target versus original-input stdlib graph, all eleven numerical stages and mass; no device execution implied",
    )
    (root / "target-application-gate.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    shutil.copyfile(__file__, root / "target-application-gate-driver.py")
    mfest = json.loads((root / "manifest.json").read_text())
    mfest["target_application_gate"] = "target-application-gate.json"
    for name in ("target-application-gate.json", "target-application-gate-driver.py"):
        mfest["files"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(mfest, indent=2) + "\n")
    assert report["passed"], f"Target preflight failed: {failure}; SDK must not run"
    return report


def device_checks(root):
    import numpy as np
    from mesh_cache_attention_sdk import unshard

    root = Path(root)
    s, m, bs, r = [
        json.loads((root / n).read_text())
        for n in ("schedule.json", "semantic.json", "batches.json", "results.json")
    ]
    assert r["success"] and len(r["cases"]) == len(r["diagnostics"]) == len(bs)
    checks = []
    for batch, outputs, d in zip(bs, r["cases"], r["diagnostics"]):
        half = lambda k: np.asarray(d[k], np.uint16).view(np.float16).astype(float)
        obs = {
            k: unshard(s, half(port), axis, width).ravel().tolist()
            for k, port, axis, width in [
                ("normalized", "normalized", "y", s["N"]),
                ("rotated_query", "Q", "x", s["N"]),
                ("score", "score", "y", s["S"]),
                ("probability", "probability", "y", s["S"]),
                ("context", "context", "x", s["N"]),
                ("delta", "delta", "y", s["N"]),
            ]
        }
        for i, name in enumerate(("query", "key_projection")):
            obs[name] = (
                unshard(
                    s,
                    half("projections")[
                        :, :, i * s["B"] * s["Nt"] : (i + 1) * s["B"] * s["Nt"]
                    ],
                    "x",
                    s["N"],
                )
                .ravel()
                .tolist()
            )
        checks.append(check(s["B"], s["N"], s["S"], batch, outputs, obs))
    return checks
