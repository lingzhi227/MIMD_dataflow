"""Register only executed resident normalized FFN profiles, with independent native/device checks."""

import argparse, datetime, json, subprocess, sys
from pathlib import Path
from probe_runtime import read, sha

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from prefill_tail_fixtures import batches, check
from native_transport import parse_outputs
from run_ports import numerical_summary

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
catalog = read(ROOT / "catalog.json")
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
    assert s["profile"] == "mesh_prefill_tail.v1"
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
        for path in (ROOT / "evidence").glob("prefill-tail-audit-mutations-*.json")
    ]
    assert any(
        v["passed"]
        and Path(v["source_bundle"]) == bundle
        and v["results_sha256"] == sha(bundle / "results.json")
        for v in mutations
    )
    from mesh_common import unpack_tiles
    import numpy as np

    def native_observed(directory, expected_node):
        observation = read(bundle / directory / "observation.json")
        assert observation["passed"] and observation["source_sha256"] == sha(
            bundle / "source.cpp"
        )
        assert observation["node"] == expected_node
        for name, digest in observation["files"].items():
            assert sha(bundle / directory / name) == digest
        observed = parse_outputs((bundle / directory / "stdout.txt").read_text())
        return [v[observation["host"]] for v in observed]

    nodes = read(bundle / "semantic.json")["nodes"]
    actual_native_projection = native_observed(
        "native-projection-observation", nodes[7]["id"]
    )
    actual_native_delta = native_observed("native-delta-observation", nodes[14]["id"])
    diagnostics = read(bundle / "results.json")["diagnostics"]

    def device_observed(key):
        return [
            unpack_tiles(
                np.asarray(v[key], np.uint16).view(np.float16), s["Mt"], s["Nt"], "F"
            )
            for v in diagnostics
        ]

    def checks(outputs, projections, deltas):
        assert len(outputs) == len(projections) == len(deltas) == len(b) == count
        return [
            check(s["M"], s["N"], s["F"], s["epsilon"], x, y, pr, d)
            for x, y, pr, d in zip(b, outputs, projections, deltas)
        ]

    case = dict(
        key="waferllm/" + bundle.parent.name,
        artifact=str(bundle.relative_to(ROOT)),
        passed=True,
        level="sdk_simulator",
        audit=audit,
        linked_static_memory=memory,
        linked_static_memory_sha256=sha(memory_path),
        native_application_checks=checks(
            native, actual_native_projection, actual_native_delta
        ),
        device_application_checks=checks(
            device,
            device_observed("projection_snapshot"),
            device_observed("down_snapshot"),
        ),
        native_application_source="native-output.txt: actual C++ stdout plus Clang-range native projection and down observations; independent original-input projection/residual/RMS/MLP/residual fsum/sqrt/exp review",
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
        fixture=f"prefill_tail:{s['M']}:{s['N']}:{s['F']}:{s['P']}",
        instrumentation=s["instrumentation"],
        status="source_ready",
        contract=f"Resident supplied attention-output tail {s['M']}x{s['N']} via {s['F']} hidden features on {s['P']}x{s['P']} PEs. Typed half output projection, residual, RMSNorm, block-f32 up/gate/down, SiLU/multiply and final postprojection Z add. Shared forward two-hop DSR engine, CSL RMS/accumulator libraries and joined completion. Immutable public inputs; no intermediate host transfer. Eight SDK calls; independent actual native/device projection, delta and final accuracy; exact staged half/f32 observations, mutation checks, ELF memory and three matched adapted-source calls. Not full Prefill/Decode or hardware performance.",
        source_comparison=str(comparison.relative_to(ROOT)),
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
