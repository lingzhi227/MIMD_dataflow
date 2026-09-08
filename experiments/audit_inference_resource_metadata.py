"""Reconcile explicit DSR lease metadata without modifying frozen SDK evidence."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import datetime, hashlib, json, sys, tempfile
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from planner import plan
from backend import generate


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def numerical_plan(v):
    if isinstance(v, dict):
        return {
            k: numerical_plan(x)
            for k, x in v.items()
            if k not in ("resources", "projection_stage")
        }
    if isinstance(v, list):
        return [numerical_plan(x) for x in v]
    return v


needed = {
    "mesh_mlp.v1",
    "mesh_device_matmul.v1",
    "mesh_score.v1",
    "mesh_score_softmax.v1",
    "mesh_attention.v1",
    "mesh_normalized_matmul.v1",
    "mesh_normalized_fanout.v1",
}
chosen = {
    "mesh_mlp.v1": ROOT
    / "benchmarks/inference/waferllm/mlp_64x64x256_8x8/run-20260907T044447648813Z"
}
for index in sorted((ROOT / "validation/evidence").glob("qualification-*.json"), reverse=True):
    for c in read(index).get("cases", []):
        p = ROOT / c.get("artifact", "missing")
        f = p / "schedule.json"
        if not c.get("passed") or not f.exists() or not (p / "results.json").exists():
            continue
        profile = read(f).get("profile")
        if profile in needed and profile not in chosen:
            chosen[profile] = p
assert set(chosen) == needed
rows = []
for profile, p in chosen.items():
    old = read(p / "schedule.json")
    new = plan(read(p / "semantic.json"))
    assert numerical_plan(old) == numerical_plan(new), profile
    with tempfile.TemporaryDirectory() as td:
        generate(new, td)
        generated = list(Path(td).iterdir())
        assert generated
        for f in generated:
            assert f.read_bytes() == (p / f.name).read_bytes(), (profile, f.name)
        emitted = {f.name: sha(f) for f in generated}
    rows.append(
        dict(
            profile=profile,
            bundle=str(p.relative_to(ROOT)),
            old_resources=old.get("resources"),
            corrected_resources=new.get("resources"),
            generated_files_sha256=emitted,
            generated_bytes_identical=True,
            arithmetic_storage_and_schedule_unchanged=True,
            additional_projection_contract_metadata="projection_stage" not in old
            and "projection_stage" in new,
            hashes={
                str(f.relative_to(ROOT)): sha(f)
                for f in (
                    p / "manifest.json",
                    p / "schedule.json",
                    p / "semantic.json",
                    p / "results.json",
                )
            },
        )
    )
output = (
    ROOT
    / "validation/evidence"
    / (
        "inference-dsr-metadata-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + ".json"
    )
)
output.write_text(
    json.dumps(
        dict(
            passed=True,
            new_sdk_execution=False,
            frozen_bundles_modified=False,
            scope="Explicit bank-specific DSR metadata correction; compiler/SDK temporary registers are not enumerated. Resource fields differ while generated files are byte-identical. Existing normalized projection plans also gain the previously introduced declarative rectangular projection contract.",
            cases=rows,
            implementation_hashes={
                str(p.relative_to(ROOT)): sha(p)
                for p in [
                    ROOT / "lib/Analysis/inference_resources.py",
                    ROOT / "runtime/csl/inference_comm.csl",
                    Path(__file__),
                ]
            },
        ),
        indent=2,
    )
    + "\n"
)
print(output.relative_to(ROOT))
