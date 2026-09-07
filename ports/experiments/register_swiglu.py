"""Register only executed gated activation profiles, with independent native/device checks."""

import argparse, datetime, json, subprocess, sys
from pathlib import Path
from probe_runtime import read, sha

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from swiglu_fixtures import batches, check
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
        "swiglu64x256-sampled-source-comparison.json",
        "swiglu64x256-counter-source-comparison.json",
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
    assert s["instrumentation"] == "sampled"
    b = read(bundle / "batches.json")
    native = parse_outputs((bundle / "native-output.txt").read_text())
    device = read(bundle / "results.json")["cases"]
    assert b == batches(s["M"], s["N"]) and len(b) == len(native) == len(device) == 6
    mutations = [
        read(path)
        for path in (ROOT / "evidence").glob("gated-activation-audit-mutations-*.json")
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
        native_application_checks=checks(native),
        device_application_checks=checks(device),
        native_application_source="native-output.txt: actual C++ stdout, independent stable math.exp review",
        qualification_sha256=sha(bundle / "qualification.json"),
        results_sha256=sha(bundle / "results.json"),
    )
    case["numerical_validation"] = numerical_summary(case)
    cases.append(case)
    item = read(bundle.parent / "PORT.json")
    item.update(
        fixture=f"gated_silu:{s['M']}:{s['N']}",
        status="source_ready",
        contract=f"Gated SiLU {s['M']}x{s['N']} on {s['rows']}x{s['cols']} PEs; six SDK calls, exact activation/product half bits, immutable inputs, phase/epoch audit and independent actual native/device standard composition. Finite magnitude <=8; component error <=.004*abs(reference)+2^-24*(1+abs(up)), not uniform relative/f32 accuracy. Isolated original source control at64x256; no full MLP/inference or hardware claim.",
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
