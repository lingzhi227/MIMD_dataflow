"""Original-input native/device stage gates for the resident composition."""

import hashlib
import json
import shutil
from pathlib import Path
from native_observers import observe
from application_gate import seal
from composed_ffn_fixtures import check
from projected_cache_ffn_reference import reference, actual_stages

ROOT = Path(__file__).resolve().parents[1]


def read(root, name):
    return json.loads((root / name).read_text())


def stage_nodes(s):
    attention = s["graph"]["attention"]["nodes"]
    ffn = s["graph"]["ffn"]["nodes"]
    result = dict(
        zip(
            (
                "normalized",
                "query",
                "key_projection",
                "value_projection",
                "rotated_query",
                "rotated_key",
                "transpose",
                "score",
                "probability",
                "context",
                "delta",
                "result",
            ),
            (n["id"] for n in attention[10:22]),
        )
    )
    result.pop("transpose")
    result.update(
        zip(
            (
                "ffn_normalized",
                "up",
                "gate",
                "activation",
                "hidden",
                "ffn_delta",
                "final_result",
            ),
            (n["id"] for n in ffn[5:12]),
        )
    )
    return result


def dimensions(s):
    return dict(b=s["B"], n=s["N"], s=s["S"], f=s["F"])


def bind_files(root, files):
    manifest = read(root, "manifest.json")
    for path in files:
        manifest["files"][str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def seal_native(root):
    root = Path(root).resolve()
    s = read(root, "schedule.json")
    mapping = stage_nodes(s)
    for name in ("final_result", "rotated_key", "value_projection"):
        mapping.pop(name)
    paths = {name: root / ("native-" + name + "-observation") for name in mapping}
    observed = {
        name: observe(root, node, paths[name])[0] for name, node in mapping.items()
    }
    rows = iter(
        [
            {name: values[e] for name, values in observed.items()}
            for e in range(s["epochs"])
        ]
    )
    report = seal(
        root,
        ROOT / "composed_ffn_fixtures.py",
        lambda batch, outputs: check(
            **dimensions(s), batch=batch, outputs=outputs, observed=next(rows)
        ),
        dimensions(s),
        native_observations=list(paths.values()),
    )
    dependencies = root / "application-dependencies"
    dependencies.mkdir()
    for name in ("projected_cache_fixtures.py", "cache_attention_fixtures.py"):
        shutil.copy2(ROOT / name, dependencies / name)
    shutil.copy2(__file__, dependencies / "composed_ffn_gate.py")
    bind_files(root, list(dependencies.iterdir()))
    return report


def seal_target(root):
    root = Path(root).resolve()
    s = read(root, "schedule.json")
    destination = root / "target-application-gate.json"
    assert not destination.exists()
    reports = []
    failure = None
    for epoch, batch in enumerate(read(root, "batches.json")):
        try:
            _, observed = reference(s, batch)
            observed = {k: v.ravel().tolist() for k, v in observed.items()}
            outputs = dict(
                result=observed["final_result"],
                new_key=observed["rotated_key"],
                new_value=observed["value_projection"],
            )
            reports.append(
                check(**dimensions(s), batch=batch, outputs=outputs, observed=observed)
            )
        except (ValueError, AssertionError) as error:
            failure = dict(epoch=epoch, error=str(error))
            break
    report = dict(
        passed=failure is None,
        checks=reports,
        failure=failure,
        scope="Predicted target vs original-input stdlib eighteen-stage mathematics and probability mass. No device observation implied.",
    )
    destination.write_text(json.dumps(report, indent=2) + "\n")
    bind_files(root, [destination])
    manifest = read(root, "manifest.json")
    manifest["target_application_gate"] = destination.name
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    assert report["passed"], "Target preflight must pass before SDK: " + str(failure)
    return report


def device_checks(root):
    root = Path(root)
    s, batches, result = [
        read(root, name) for name in ("schedule.json", "batches.json", "results.json")
    ]
    assert (
        result["success"] is True
        and len(result["cases"])
        == len(result["diagnostics"])
        == len(batches)
        == s["epochs"]
    )
    return [
        check(
            **dimensions(s),
            batch=batch,
            outputs=output,
            observed={k: v.ravel().tolist() for k, v in actual_stages(s, raw).items()}
        )
        for batch, output, raw in zip(batches, result["cases"], result["diagnostics"])
    ]
