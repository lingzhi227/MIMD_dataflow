"""Admit an executed mixed input-attention profile only after all evidence gates.

Registration does not execute the SDK, change frozen runs, or claim hardware
performance. All comparisons and mutation reviews must bind these exact results.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from probe_runtime import read, sha

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from input_attention_fixtures import batches
from input_attention_mixed_gate import device_checks
from run_profiles import numerical_summary


def register(bundle, comparison, memory, mutations, host_native):
    bundle, comparison, memory, mutations, host_native = map(
        lambda x: Path(x).resolve(), (bundle, comparison, memory, mutations, host_native)
    )
    catalog = read(ROOT / "benchmarks/catalog.json")
    item = read(bundle.parent / "PORT.json")
    key = (item["project"], item["kernel"])
    assert key not in {(x["project"], x["kernel"]) for x in catalog}
    assert read(bundle / "qualification.json")["success"]
    code = ('import json,sys;from pathlib import Path;p=Path(sys.argv[1]);'
            'sys.path.insert(0,str(p/"implementation"));'
            'from integrity import verify_bundle,verify_codegen;'
            'verify_bundle(p);verify_codegen(p);'
            'from validate import audit;print(json.dumps(audit(p)))')
    audit = json.loads(subprocess.check_output([sys.executable, "-c", code, str(bundle)], text=True))
    assert audit["passed"]
    s = read(bundle / "schedule.json")
    assert s["profile"] == "mesh_input_attention_mixed.v1"
    assert [s[k] for k in ("M", "N", "F", "P")] == [64, 64, 256, 8]
    assert read(bundle / "batches.json") == batches(64, 64, 256, 8)
    c, mem, fault = map(read, (comparison, memory, mutations))
    assert c["passed"] and c["precision_matched"] and c["source_mathematics_eight_passed"]
    assert len(c["cases"]) == 8 and all(v["exact_port_groups"] == 23 for v in c["cases"])
    for path, digest in c["hashes"].items():
        assert sha(ROOT / path) == digest, path
    for name in ("manifest.json", "results.json", "qualification.json"):
        assert c["hashes"][str((bundle / name).relative_to(ROOT))] == sha(bundle / name)
    assert mem["classes"] and mem["max_static_high_water_bytes"] <= 49152
    for row in mem["classes"]:
        assert sha(bundle / "out/bin" / row["elf"]) == row["sha256"]
    assert fault["passed"] and fault["mutations"] >= 85
    assert Path(fault["source_bundle"]) == bundle
    for name in ("results", "manifest"):
        assert fault[name + "_sha256"] == sha(bundle / (name + ".json"))
    assert fault["driver_sha256"] == sha(ROOT / "experiments/audit_input_attention_mixed_mutations.py")
    hn = read(host_native / "review.json")
    assert hn["passed"] and hn["epochs"] == 8 and hn["observed_branches"] == 13
    assert hn["generated_csl_identical"] and hn["prior_manifest_sha256"] == sha(bundle / "manifest.json")
    for path, digest in read(host_native / "provenance.json")["files"].items():
        assert sha(host_native / path) == digest, path
    native = read(bundle / "application-gate.json")
    assert native["passed"] and len(native["checks"]) == 8
    assert len(native["native_observations"]) == 13
    case = dict(key="/".join(key), artifact=str(bundle.relative_to(ROOT)),
                passed=True, level="sdk_simulator", audit=audit,
                native_application_checks=native["checks"],
                device_application_checks=device_checks(bundle),
                linked_static_memory=mem, linked_static_memory_sha256=sha(memory),
                qualification_sha256=sha(bundle / "qualification.json"),
                results_sha256=sha(bundle / "results.json"),
                source_comparison=str(comparison.relative_to(ROOT)),
                performance=dict(max_ratio_of_max_pe_cycles=c["max_ratio"],
                                 min_ratio_of_max_pe_cycles=c["min_ratio"],
                                 scope=c["timing_scope"]))
    case["numerical_validation"] = numerical_summary(case)
    assert case["numerical_validation"]["fixed_accuracy_passed"]
    item.update(status="source_ready", fixture="input_attention_mixed:64:64:256:8",
                target_replay="validation/evidence/input-attention-codegen-20260907T142401616614Z",
                source_comparison=str(comparison.relative_to(ROOT)),
                qualification="Eight standard-driver SDK calls; original-eleven-input branch and final accuracy, actual native observers on two hosts, exact matched-source data, frozen audit mutations, linked ELF memory and simulator cycle comparison. Scoped single-head unmasked chain, not full Prefill/Decode or hardware performance.")
    report = dict(kind="qualification_index_of_existing_sdk_runs", sdk=True,
                  success=True, new_sdk_execution=False, cases=[case],
                  source_controls=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p))
                                   for p in (comparison, memory, mutations, host_native / "provenance.json", host_native / "review.json", Path(__file__))])
    path = ROOT / "validation/evidence" / ("qualification-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    assert not path.exists()
    path.write_text(json.dumps(report, indent=2) + "\n")
    (bundle.parent / "PORT.json").write_text(json.dumps(item, indent=2) + "\n")
    catalog.append(item)
    (ROOT / "benchmarks/catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(path.relative_to(ROOT), "catalog", len(catalog))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "comparison", "memory", "mutations", "host_native"):
        p.add_argument(name, type=Path)
    register(**vars(p.parse_args()))
