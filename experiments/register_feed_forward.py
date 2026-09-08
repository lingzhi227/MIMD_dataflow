"""Register only executed resident normalized FFN profiles, with independent native/device checks."""

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
from feed_forward_fixtures import batches, check
from native_transport import parse_outputs
from run_profiles import numerical_summary

p = argparse.ArgumentParser()
p.add_argument("bundles", type=Path, nargs="+")
p.add_argument("--comparison", type=Path)
p.add_argument("--memory", type=Path)
a = p.parse_args()
if a.comparison:
    a.comparison = a.comparison.resolve()
if a.memory:
    a.memory = a.memory.resolve()
assert not (a.comparison or a.memory) or len(a.bundles) == 1
catalog = read(ROOT / "benchmarks/catalog.json")
seen = {(x["project"], x["kernel"]) for x in catalog}
controls = []
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
    assert s["profile"] == "mesh_feed_forward.v1"
    count = 8
    comparison = a.comparison
    assert (
        comparison is not None and a.memory is not None
    ), "explicit matched source and ELF reports required"
    c = read(comparison)
    assert c["passed"] and c["blocked_upper_accumulation"]
    assert c["hashes"][str(bundle.relative_to(ROOT) / "manifest.json")] == sha(
        bundle / "manifest.json"
    )
    for path, digest in c["hashes"].items():
        assert sha(ROOT / path) == digest, "matched comparison dependency changed"
    assert c["hashes"][str(bundle.relative_to(ROOT) / "results.json")] == sha(
        bundle / "results.json"
    )
    assert (
        c["source_math_review"]["passed"]
        and not c["source_math_review"]["preflight_only"]
    )
    for path, digest in c["source_math_review"]["hashes"].items():
        assert (
            sha(ROOT / path) == digest
        ), "source mathematical review dependency changed"
    controls.append(comparison)
    memory_path = a.memory
    memory = read(memory_path)
    assert memory["max_static_high_water_bytes"] <= 49152
    assert memory["classes"]
    for item in memory["classes"]:
        assert sha(bundle / "out/bin" / item["elf"]) == item["sha256"]
    b = read(bundle / "batches.json")
    native = parse_outputs((bundle / "native-output.txt").read_text())
    device = read(bundle / "results.json")["cases"]
    assert (
        b == batches(s["M"], s["N"], s["F"], s["P"])
        and len(b) == len(native) == len(device) == count
    )
    mutations = [
        read(path)
        for path in (ROOT / "validation/evidence").glob("feed-forward-audit-mutations-*.json")
    ]
    assert any(
        v["passed"]
        and Path(v["source_bundle"]) == bundle
        and v["results_sha256"] == sha(bundle / "results.json")
        for v in mutations
    )
    from mesh_common import unpack_tiles
    import numpy as np

    observation = read(bundle / "native-observation/observation.json")
    assert observation["passed"] and observation["source_sha256"] == sha(
        bundle / "source.cpp"
    )
    observed_native = parse_outputs(
        (bundle / "native-observation/stdout.txt").read_text()
    )
    actual_native_delta = [v[observation["host"]] for v in observed_native]
    diagnostics = read(bundle / "results.json")["diagnostics"]
    actual_device_delta = [
        unpack_tiles(
            np.asarray(v["down_snapshot"], np.uint16).view(np.float16),
            s["Mt"],
            s["Nt"],
            "F",
        )
        for v in diagnostics
    ]

    def checks(outputs, deltas):
        assert len(outputs) == len(deltas) == len(b) == count
        return [
            check(s["M"], s["N"], s["F"], s["epsilon"], x, y, d)
            for x, y, d in zip(b, outputs, deltas)
        ]

    case = dict(
        key="waferllm/" + bundle.parent.name,
        artifact=str(bundle.relative_to(ROOT)),
        passed=True,
        level="sdk_simulator",
        audit=audit,
        linked_static_memory=memory,
        linked_static_memory_sha256=sha(memory_path),
        native_application_checks=checks(native, actual_native_delta),
        device_application_checks=checks(device, actual_device_delta),
        native_application_source="native-output.txt: actual C++ stdout plus Clang-range native down observation; independent original-input RMS/MLP/residual fsum/sqrt/exp review",
        qualification_sha256=sha(bundle / "qualification.json"),
        results_sha256=sha(bundle / "results.json"),
    )
    assert all(
        v["fixed_accuracy_passed"]
        for v in case["native_application_checks"] + case["device_application_checks"]
    )
    case["numerical_validation"] = numerical_summary(case)
    cases.append(case)
    item = read(bundle.parent / "PORT.json")
    item.update(
        fixture=f"feed_forward:{s['M']}:{s['N']}:{s['F']}:{s['P']}",
        instrumentation=s["instrumentation"],
        status="source_ready",
        contract=f"Resident supplied Z/gamma/up/gate/down weights: {s['M']}x{s['N']} normalized gated FFN via {s['F']} hidden features on {s['P']}x{s['P']} PEs. Typed RMSNorm/matmul/SiLU/multiply/add graph with explicit block-f32 all projections, source forward two-hop DSR engine, original row collective and reusable CSL RMS/accumulator libraries. Immutable inputs and no intermediate host transfer. Eight SDK calls, exact normalized/delta/output and three f32 accumulator observations; independent actual native/device delta and final accuracy, mutations and ELF memory. Matched repaired-source comparison first three calls with observer/copy differences stated. Not full Prefill/Decode or hardware performance.",
        source_comparison=str(comparison.relative_to(ROOT)),
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
