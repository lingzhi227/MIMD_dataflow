"""Register only executed normalized fan-out profiles, with independent native/device checks."""

import argparse, datetime, json, subprocess, sys
from pathlib import Path
from probe_runtime import read, sha

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from normalized_fanout_fixtures import batches, check
from native_transport import parse_outputs
from run_ports import numerical_summary

p = argparse.ArgumentParser()
p.add_argument("bundles", type=Path, nargs="+")
a = p.parse_args()
catalog = read(ROOT / "catalog.json")
seen = {(x["project"], x["kernel"]) for x in catalog}
controls = [
    ROOT / "evidence" / x
    for x in [
        "normalized-fanout64x128-counter-source-comparison.json",
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
    b = read(bundle / "batches.json")
    native = parse_outputs((bundle / "native-output.txt").read_text())
    device = read(bundle / "results.json")["cases"]
    assert (
        b == batches(s["M"], s["N"], s["projections"])
        and len(b) == len(native) == len(device) == 6
    )
    mutations = [
        read(path)
        for path in (ROOT / "evidence").glob("normalized-fanout-audit-mutations-*.json")
    ]
    assert any(
        v["passed"]
        and Path(v["source_bundle"]) == bundle
        and v["results_sha256"] == sha(bundle / "results.json")
        for v in mutations
    )
    checks = lambda outputs: [
        check(s["M"], s["N"], s["projections"], x, y) for x, y in zip(b, outputs)
    ]
    case = dict(
        key="waferllm/" + bundle.parent.name,
        artifact=str(bundle.relative_to(ROOT)),
        passed=True,
        level="sdk_simulator",
        audit=audit,
        native_application_checks=checks(native),
        device_application_checks=checks(device),
        native_application_source="native-output.txt: actual C++ stdout, independent math.fsum RMS and dense branch review",
        qualification_sha256=sha(bundle / "qualification.json"),
        results_sha256=sha(bundle / "results.json"),
    )
    case["numerical_validation"] = numerical_summary(case)
    cases.append(case)
    item = read(bundle.parent / "PORT.json")
    item.update(
        fixture=f"normalized_fanout:{s['M']}:{s['N']}:{s['projections']}",
        instrumentation=s["instrumentation"],
        status="source_ready",
        contract=f"Resident RMSNorm plus {s['projections']} independent square projections {s['M']}x{s['N']} on {s['P']}x{s['P']} PEs. Shared normalization/alignment and live buffer ownership, sequential branches with overlapped per-branch two-hop communication and local DSR math. Six SDK calls including independent changed branch weights, exact final half bits, sampled prefixes/live ownership where enabled, immutable inputs, epochs/queues and actual native/device independent math.fsum review. Half normwise .015 L2/.02 peak; inputs<=1. Counter source comparison is scoped to three projections64x128/8x8 with explicit source repairs; no performance extrapolation to other geometries, full inference or hardware.",
    )
    additions.append((bundle.parent, item))
path = (
    ROOT
    / "evidence"
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
(ROOT / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
print(path.relative_to(ROOT), len(additions))
