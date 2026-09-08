"""Register one measured Decode-layout transform after full source/native/ELF/fault gates."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, subprocess, sys
from pathlib import Path
from probe_runtime import read, sha, SDK_HASH

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from native_transport import parse_outputs
from pair_rotation_fixtures import batches, check
from run_profiles import numerical_summary

p = argparse.ArgumentParser()
for name in ("bundle", "comparison", "memory", "mutations", "host"):
    p.add_argument(name, type=Path)
a = p.parse_args()
bundle, comparison, memory, mutations, host = [
    v.resolve() for v in (a.bundle, a.comparison, a.memory, a.mutations, a.host)
]
item = read(bundle.parent / "PORT.json")
catalog = read(ROOT / "benchmarks/catalog.json")
assert (item["project"], item["kernel"]) not in {
    (v["project"], v["kernel"]) for v in catalog
}
q = read(bundle / "qualification.json")
assert q["success"] and q["sdk_sha256"] == SDK_HASH
code = 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
audit = json.loads(
    subprocess.check_output([sys.executable, "-c", code, str(bundle)], text=True)
)
assert audit["passed"] and audit["epochs"] == 6
s = read(bundle / "schedule.json")
assert (s["M"], s["N"], s["rows"], s["cols"], s["axis"], s["layout"]) == (
    5,
    1024,
    8,
    8,
    "x",
    "batch_major",
)
bs = read(bundle / "batches.json")
native = parse_outputs((bundle / "native-output.txt").read_text())
device = read(bundle / "results.json")["cases"]
hnative = parse_outputs((host / "build/native-output.txt").read_text())
assert (
    bs == batches(5, 1024, True, "odd_even")
    and len(bs) == len(native) == len(device) == len(hnative) == 6
)
c = read(comparison)
assert c["passed"] and c["epochs"] == len(c["checks"]) == 6
for name, digest in c["files"].items():
    assert sha(ROOT / name) == digest, name
assert c["files"][str((bundle / "manifest.json").relative_to(ROOT))] == sha(
    bundle / "manifest.json"
)
mem = read(memory)
assert mem["max_static_high_water_bytes"] <= min(
    49152, sum(s["memory_per_pe"].values())
) and mem["elf_classes"] == len(mem["classes"])
for row in mem["classes"]:
    assert sha(bundle / "out/bin" / row["elf"]) == row["sha256"]
fault = read(mutations)
assert (
    fault["passed"]
    and fault["mutations"] == len(fault["checks"]) >= 30
    and Path(fault["source_bundle"]) == bundle
)
assert (
    fault["manifest_sha256"] == sha(bundle / "manifest.json")
    and fault["results_sha256"] == sha(bundle / "results.json")
    and fault["driver_sha256"]
    == sha(ROOT / "experiments/audit_batched_pair_mutations.py")
)
hn = read(host / "review.json")
assert (
    hn["passed"]
    and hn["csl_identical"]
    and hn["prior_manifest_sha256"] == sha(bundle / "manifest.json")
)
for name, digest in read(host / "provenance.json")["files"].items():
    assert sha(host / name) == digest, name
for f in (host / "build").glob("*.csl"):
    assert f.read_bytes() == (bundle / f.name).read_bytes()
checks = lambda outputs: [
    check(5, 1024, True, "odd_even", b, o) for b, o in zip(bs, outputs)
]
case = dict(
    key="waferllm/" + item["kernel"],
    artifact=str(bundle.relative_to(ROOT)),
    passed=True,
    level="sdk_simulator",
    audit=audit,
    native_application_source="Actual C++ stdout on two hosts; independent original-input math.fsum component checks",
    native_application_checks=checks(native),
    sdk_host_native_application_checks=checks(hnative),
    device_application_checks=checks(device),
    linked_static_memory=mem,
    qualification_sha256=sha(bundle / "qualification.json"),
    results_sha256=sha(bundle / "results.json"),
)
case["numerical_validation"] = numerical_summary(case)
for name in (
    "decode-pair-original-offset-failure.json",
    "dsd-base-offset-20260907T231325901420Z/offset-review.json",
):
    assert read(ROOT / "validation/evidence" / name)["passed"]
controls = [
    ROOT / "validation/evidence/decode-pair-original-offset-failure.json",
    ROOT / "validation/evidence/dsd-base-offset-20260907T231325901420Z/offset-review.json",
    comparison,
    memory,
    mutations,
    host / "review.json",
    host / "provenance.json",
    ROOT / "tests/support/pair_rotation_fixtures.py",
    Path(__file__),
]
out = (
    ROOT
    / "validation/evidence"
    / (
        "qualification-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + ".json"
    )
)
assert not out.exists()
item.update(
    status="source_ready",
    contract="Decode-layout adjacent pair transform B5/N1024 on8x8, feature X partition/Y replicas, source odd_even order and supplied shared coefficients. Six matched SDK/source calls; all PE input/result/products exact, component mathematics on two native hosts and device, fault rejection and ELF checks. Source control explicitly repairs the odd DSD offset after base reset and includes immutable input copy. Unrepaired-source failure and actual DSD probe retained. No full model, position generation, cache update or hardware throughput claim.",
)
out.write_text(
    json.dumps(
        dict(
            kind="qualification_index_of_existing_sdk_runs",
            sdk=True,
            success=True,
            new_sdk_execution=False,
            cases=[case],
            source_controls=[
                dict(path=str(v.relative_to(ROOT)), sha256=sha(v)) for v in controls
            ],
        ),
        indent=2,
    )
    + "\n"
)
(bundle.parent / "PORT.json").write_text(json.dumps(item, indent=2) + "\n")
catalog.append(item)
(ROOT / "benchmarks/catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
print("BATCHED PAIR REGISTERED", len(catalog), out)
