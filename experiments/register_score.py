"""Register only executed score contraction profiles, with independent native/device checks."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, subprocess, sys
from pathlib import Path
from probe_runtime import read, sha

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from score_fixtures import batches, check
from native_transport import parse_outputs
from run_profiles import numerical_summary

p = argparse.ArgumentParser()
p.add_argument("bundles", type=Path, nargs="+")
a = p.parse_args()
catalog = read(ROOT / "benchmarks/catalog.json")
seen = {(x["project"], x["kernel"]) for x in catalog}
controls = [
    ROOT / "validation/evidence" / x
    for x in [
        "score64x128-counter-source-comparison.json",
    ]
]
for control in controls:
    assert read(control)["passed"]
cases = []
additions = []
for bundle in a.bundles:
    bundle = bundle.resolve()
    assert ("waferllm", bundle.parent.name) not in seen
    seen.add(("waferllm", bundle.parent.name))
    q = read(bundle / "qualification.json")
    assert q["success"]
    code = 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
    audit = json.loads(
        subprocess.check_output([sys.executable, "-c", code, str(bundle)], text=True)
    )
    assert audit["passed"]
    s = read(bundle / "schedule.json")
    comparison = (
        ROOT
        / "validation/evidence"
        / f"score{s['M']}x{s['N']}-{s['instrumentation']}-source-comparison.json"
    )
    c = read(comparison)
    assert c["passed"]
    assert c["hashes"][str(bundle.relative_to(ROOT) / "manifest.json")] == sha(
        bundle / "manifest.json"
    )
    controls.append(comparison)
    memory_path = ROOT / "validation/evidence" / f"score{s['M']}x{s['N']}-static-memory.json"
    memory = read(memory_path)
    assert memory["max_static_high_water_bytes"] <= 49152
    assert memory["classes"]
    for item in memory["classes"]:
        assert sha(bundle / "out/bin" / item["elf"]) == item["sha256"]
    b = read(bundle / "batches.json")
    native = parse_outputs((bundle / "native-output.txt").read_text())
    device = read(bundle / "results.json")["cases"]
    assert b == batches(s["M"], s["N"]) and len(b) == len(native) == len(device) == 6
    mutations = [
        read(path) for path in (ROOT / "validation/evidence").glob("score-audit-mutations-*.json")
    ]
    assert any(
        v["passed"]
        and Path(v["source_bundle"]) == bundle
        and v["results_sha256"] == sha(bundle / "results.json")
        for v in mutations
    )
    checks = lambda outputs: [check(s["M"], s["N"], x, y) for x, y in zip(b, outputs)]
    case = dict(
        key="waferllm/" + bundle.parent.name,
        artifact=str(bundle.relative_to(ROOT)),
        passed=True,
        level="sdk_simulator",
        audit=audit,
        linked_static_memory=memory,
        linked_static_memory_sha256=sha(memory_path),
        native_application_checks=checks(native),
        device_application_checks=checks(device),
        native_application_source="native-output.txt: actual C++ stdout, independent math.fsum original-input QK-transpose review",
        qualification_sha256=sha(bundle / "qualification.json"),
        results_sha256=sha(bundle / "results.json"),
    )
    case["numerical_validation"] = numerical_summary(case)
    cases.append(case)
    item = read(bundle.parent / "PORT.json")
    item.update(
        fixture=f"score:{s['M']}:{s['N']}",
        instrumentation=s["instrumentation"],
        status="source_ready",
        contract=f"QK-transpose {s['M']}x{s['N']} on {s['P']}x{s['P']} PEs, typed transpose view and original vertical K exchange/horizontal EAST-first rotating-root reduction. Six SDK calls with changed dense Q/K, immutable inputs, exact target half outputs and sampled per-round partial/live K ownership, root/epoch/queue audits. Actual native/device independent math.fsum checks under .015L2/.02peak normwise policy, input<=1. Small counter control measures local simulator overhead separately from tensor observations; larger sampled comparison includes unequal live-owner observer costs. Not full attention or hardware performance.",
    )
    additions.append((bundle.parent, item))
path = (
    ROOT
    / "validation/evidence"
    / (
        "qualification-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + ".json"
    )
)
assert not path.exists()
path.write_text(
    json.dumps(
        dict(
            kind="qualification_index_of_existing_sdk_runs",
            sdk=True,
            success=True,
            new_sdk_execution=False,
            source_controls=[
                dict(path=str(v.relative_to(ROOT)), sha256=sha(v)) for v in controls
            ],
            cases=cases,
        ),
        indent=2,
    )
    + "\n"
)
for folder, item in additions:
    (folder / "PORT.json").write_text(json.dumps(item, indent=2) + "\n")
    catalog.append(item)
(ROOT / "benchmarks/catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
print(path.relative_to(ROOT), len(additions))
