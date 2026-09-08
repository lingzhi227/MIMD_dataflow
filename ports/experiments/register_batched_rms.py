"""Register one executed batched RMS profile only after independent evidence gates."""

import argparse, datetime, json, subprocess, sys
from pathlib import Path
from probe_runtime import read, sha

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from batched_rms_fixtures import batches, check
from native_transport import parse_outputs
from run_ports import numerical_summary


def register(bundle, comparison, memory, mutations, host):
    bundle, comparison, memory, mutations, host = map(
        lambda p: Path(p).resolve(), (bundle, comparison, memory, mutations, host)
    )
    catalog = read(ROOT / "catalog.json")
    item = read(bundle.parent / "PORT.json")
    assert (item["project"], item["kernel"]) not in {
        (v["project"], v["kernel"]) for v in catalog
    }
    q = read(bundle / "qualification.json")
    assert q["success"]
    code = 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
    audit = json.loads(
        subprocess.check_output([sys.executable, "-c", code, str(bundle)], text=True)
    )
    assert audit["passed"]
    s = read(bundle / "schedule.json")
    bs = read(bundle / "batches.json")
    assert (
        s["profile"] == "mesh_batched_rms.v1"
        and bs == batches(s["B"], s["N"])
        and len(bs) == 8
    )
    c = read(comparison)
    assert c["passed"] and c["source_mathematics_eight_passed"] and len(c["cases"]) == 8
    for p, digest in c["hashes"].items():
        assert sha(ROOT / p) == digest, p
    for name in ("manifest.json", "results.json", "qualification.json"):
        assert c["hashes"][str((bundle / name).relative_to(ROOT))] == sha(bundle / name)
    mem = read(memory)
    assert mem["classes"] and mem["max_static_high_water_bytes"] <= 49152
    for row in mem["classes"]:
        assert sha(bundle / "out/bin" / row["elf"]) == row["sha256"]
    fault = read(mutations)
    assert (
        fault["passed"]
        and fault["mutations"] >= 28
        and Path(fault["source_bundle"]) == bundle
    )
    assert fault["results_sha256"] == sha(bundle / "results.json") and fault[
        "manifest_sha256"
    ] == sha(bundle / "manifest.json")
    assert fault["driver_sha256"] == sha(
        ROOT / "experiments/audit_batched_rms_mutations.py"
    )
    hn = read(host / "review.json")
    assert (
        hn["passed"]
        and hn["csl_identical"]
        and hn["prior_manifest_sha256"] == sha(bundle / "manifest.json")
    )
    for p, digest in read(host / "provenance.json")["files"].items():
        assert sha(host / p) == digest, p
    outputs = parse_outputs((bundle / "native-output.txt").read_text())
    device = read(bundle / "results.json")["cases"]
    assert len(outputs) == len(device) == 8
    native_checks = [check(s["B"], s["N"], b, o) for b, o in zip(bs, outputs)]
    device_checks = [check(s["B"], s["N"], b, o) for b, o in zip(bs, device)]
    case = dict(
        key=item["project"] + "/" + item["kernel"],
        artifact=str(bundle.relative_to(ROOT)),
        passed=True,
        level="sdk_simulator",
        audit=audit,
        native_application_checks=native_checks,
        device_application_checks=device_checks,
        linked_static_memory=mem,
        linked_static_memory_sha256=sha(memory),
        qualification_sha256=sha(bundle / "qualification.json"),
        results_sha256=sha(bundle / "results.json"),
        source_comparison=str(comparison.relative_to(ROOT)),
        performance=dict(
            min_ratio=c["min_ratio"], max_ratio=c["max_ratio"], scope=c["scope"]
        ),
    )
    case["numerical_validation"] = numerical_summary(case)
    assert case["numerical_validation"]["fixed_accuracy_passed"]
    item.update(
        status="source_ready",
        source_comparison=str(comparison.relative_to(ROOT)),
        qualification="Eight standard SDK calls, same-host C++ on two hosts, independent RMS accuracy, exact source local/reduced/results and replicas, odd batch padding, 28 mutation rejections, linked ELF and scoped simulator cycle comparison.",
    )
    controls = [
        comparison,
        memory,
        mutations,
        host / "review.json",
        host / "provenance.json",
        Path(__file__),
    ]
    out = (
        ROOT
        / "evidence"
        / (
            "qualification-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + ".json"
        )
    )
    assert not out.exists()
    out.write_text(
        json.dumps(
            dict(
                kind="qualification_index_of_existing_sdk_runs",
                sdk=True,
                success=True,
                new_sdk_execution=False,
                cases=[case],
                source_controls=[
                    dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in controls
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    (bundle.parent / "PORT.json").write_text(json.dumps(item, indent=2) + "\n")
    catalog.append(item)
    (ROOT / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(out, len(catalog))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("bundle", "comparison", "memory", "mutations", "host"):
        p.add_argument(name, type=Path)
    register(**vars(p.parse_args()))
