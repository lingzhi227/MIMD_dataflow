"""Seal six actual native FFN stages and a separate predicted target preflight."""

import hashlib, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain"), str(ROOT / "experiments")]
from native_observers import observe
from application_gate import seal
from batched_ffn_fixtures import check
from mesh_batched_feed_forward_sdk import values
from batched_ffn_reference import reference

INDICES = dict(normalized=5, up=6, gate=7, activation=8, hidden=9, delta=10)


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
        ROOT / "batched_ffn_fixtures.py",
        lambda b, o: check(s["B"], s["N"], s["F"], b, o, next(rows)),
        dict(B=s["B"], N=s["N"], F=s["F"]),
        native_observations=list(paths.values()),
    )


def seal_target(root):
    root = Path(root).resolve()
    assert not (root / "target-application-gate.json").exists()
    m = json.loads((root / "semantic.json").read_text())
    s = json.loads((root / "schedule.json").read_text())
    bs = json.loads((root / "batches.json").read_text())
    checks = []
    for b in bs:
        _, stages = reference(s, values(m, b))
        out = {"result": stages.pop("result").ravel().tolist()}
        checks.append(check(s["B"], s["N"], s["F"], b, out, stages))
    report = dict(
        passed=True,
        checks=checks,
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
    return report
