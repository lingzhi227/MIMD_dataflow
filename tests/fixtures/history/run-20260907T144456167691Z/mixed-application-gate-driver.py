"""Actual native branches and a pinned prior-SDK replay preflight for the mixed chain."""

import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from native_observers import observe
from application_gate import seal
from input_attention_fixtures import check
from mesh_common import unpack_tiles
from integrity import verify_bundle

INDICES = dict(
    input_normalized=11,
    q_raw=12,
    k_raw=13,
    v_raw=14,
    q=15,
    k=16,
    score=18,
    probability=19,
    attention=20,
    projection=21,
    z=22,
    normalized_z=23,
    delta=28,
)


def seal_native(root):
    root = Path(root).resolve()
    m = json.loads((root / "semantic.json").read_text())
    s = json.loads((root / "schedule.json").read_text())
    paths = [root / ("native-" + name + "-observation") for name in INDICES]
    observed = {
        name: observe(root, m["nodes"][idx]["id"], path)[0]
        for (name, idx), path in zip(INDICES.items(), paths)
    }
    rows = iter([{k: v[e] for k, v in observed.items()} for e in range(m["epochs"])])

    def checked(b, o):
        obs = next(rows)
        obs["v"] = obs["v_raw"]
        return check(s["M"], s["N"], s["F"], s["epsilon"], s["scale"], b, o, obs)

    result = seal(
        root,
        ROOT / "input_attention_fixtures.py",
        checked,
        dict(M=s["M"], N=s["N"], F=s["F"]),
        native_observations=paths,
    )
    manifest = json.loads((root / "manifest.json").read_text())
    for name in (
        "input_attention_fixtures.py",
        "attention_tail_fixtures.py",
        "prefill_tail_fixtures.py",
        "feed_forward_fixtures.py",
    ):
        shutil.copyfile(ROOT / name, root / name)
        manifest["files"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    shutil.copyfile(__file__, root / "mixed-application-gate-driver.py")
    manifest["files"]["mixed-application-gate-driver.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result


def seal_replay(root, prior):
    """Prior actual SDK data is explicitly a replay preflight, not a prediction."""
    from probe_runtime import verify as verify_probe

    root, prior = Path(root).resolve(), Path(prior).resolve()
    verify_bundle(root)
    verify_probe(prior)
    assert not (root / "target-application-gate.json").exists()
    assert (root / "source.cpp").read_bytes() == (prior / "source.cpp").read_bytes()
    for f in root.glob("*.csl"):
        assert f.read_bytes() == (prior / f.name).read_bytes(), f.name
    bs = json.loads((root / "batches.json").read_text())
    assert bs == json.loads((prior / "logical-inputs.json").read_text())
    old = json.loads((prior / "results.json").read_text())
    assert old["success"] and len(old["cases"]) == len(bs) == 8
    checks = []
    for b, row in zip(bs, old["cases"]):

        def value(name, wide=False):
            return unpack_tiles(
                np.asarray(row[name], np.uint32 if wide else np.uint16).view(
                    np.float32 if wide else np.float16
                ),
                8,
                8,
                "F",
            ).astype(float)

        obs = {
            k: value(v)
            for k, v in dict(
                input_normalized="input_normalized",
                q_raw="input_q_raw",
                k_raw="input_k_raw",
                q="x",
                k="attention_k",
                score="attention_logits",
                delta="down_snapshot",
            ).items()
        }
        obs.update(
            {
                k: value(v, True)
                for k, v in dict(
                    v_raw="mixed_v",
                    v="mixed_v",
                    probability="mixed_probability_snapshot",
                    attention="mixed_a",
                    projection="mixed_projection",
                ).items()
            }
        )
        checks.append(
            check(
                64,
                64,
                256,
                1e-6,
                0.125,
                b,
                {"output": value("result").ravel().tolist()},
                obs,
            )
        )
    report = dict(
        passed=True,
        kind="actual_prior_sdk_replay",
        checks=checks,
        prior=str(prior),
        scope="Preflight from prior actual SDK execution with identical HLS source, generated CSL and all eight original-input fixtures; independently rechecked mathematical branches. This is not predicted target arithmetic and does not replace the new standard-driver SDK run.",
        prior_results_sha256=hashlib.sha256(
            (prior / "results.json").read_bytes()
        ).hexdigest(),
        prior_provenance_sha256=hashlib.sha256(
            (prior / "provenance.json").read_bytes()
        ).hexdigest(),
    )
    (root / "target-application-gate.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    shutil.copyfile(prior / "results.json", root / "prior-sdk-replay-results.json")
    shutil.copyfile(__file__, root / "target-application-gate-driver.py")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["target_application_gate"] = "target-application-gate.json"
    for name in (
        "target-application-gate.json",
        "prior-sdk-replay-results.json",
        "target-application-gate-driver.py",
    ):
        manifest["files"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return report


if __name__ == "__main__":
    seal_native(sys.argv[1])
    seal_replay(sys.argv[1], sys.argv[2])
    print("PASS native 13 branches and actual prior-SDK replay preflight")
