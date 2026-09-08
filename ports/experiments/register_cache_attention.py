"""Admit one full SDK cache attention only after independent stage and source-control gates."""

import argparse, datetime, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from native_transport import parse_outputs
from cache_attention_fixtures import batches, check
from run_ports import numerical_summary

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def register(bundle, comparison, memory, mutations, host):
    bundle, comparison, memory, mutations, host = [
        Path(p).resolve() for p in (bundle, comparison, memory, mutations, host)
    ]
    item = read(bundle.parent / "PORT.json")
    catalog = read(ROOT / "catalog.json")
    assert (item["project"], item["kernel"]) not in {
        (v["project"], v["kernel"]) for v in catalog
    }
    q = read(bundle / "qualification.json")
    assert (
        q["success"]
        and q["sdk_sha256"]
        == "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"
    )
    code = 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
    audit = json.loads(
        subprocess.check_output([sys.executable, "-c", code, str(bundle)], text=True)
    )
    assert audit["passed"] and audit["epochs"] == 8
    s = read(bundle / "schedule.json")
    m = read(bundle / "semantic.json")
    bs = read(bundle / "batches.json")
    result = read(bundle / "results.json")
    assert s["profile"] == "mesh_cache_attention.v1" and (
        s["B"],
        s["N"],
        s["S"],
    ) == (5, 256, 512)
    assert bs == batches(5, 256, 512) and len(bs) == len(result["cases"]) == 8
    assert type(result["runtime_instances"]) is int and result["runtime_instances"] == 1
    c = read(comparison)
    assert c["passed"] and c["epochs"] == len(c["checks"]) == 8
    for name, digest in c["files"].items():
        assert sha(ROOT / name) == digest, name
    for name in ("manifest.json", "results.json", "qualification.json"):
        assert c["files"][str((bundle / name).relative_to(ROOT))] == sha(bundle / name)
    mem = read(memory)
    assert (
        mem["elf_classes"] == len(mem["classes"])
        and mem["max_static_high_water_bytes"] <= 49152
    )
    for row in mem["classes"]:
        assert sha(bundle / "out/bin" / row["elf"]) == row["sha256"]
    fault = read(mutations)
    assert (
        fault["passed"]
        and fault["mutations"] >= 40
        and Path(fault["source_bundle"]) == bundle
    )
    assert fault["results_sha256"] == sha(bundle / "results.json") and fault[
        "manifest_sha256"
    ] == sha(bundle / "manifest.json")
    assert fault["driver_sha256"] == sha(
        ROOT / "experiments/audit_cache_attention_mutations.py"
    )
    hn = read(host / "review.json")
    assert (
        hn["passed"]
        and hn["csl_identical"]
        and hn["prior_manifest_sha256"] == sha(bundle / "manifest.json")
    )
    for name, digest in read(host / "provenance.json")["files"].items():
        assert sha(host / name) == digest, name
    for p in (host / "build").glob("*.csl"):
        assert p.read_bytes() == (bundle / p.name).read_bytes()
    native = parse_outputs((bundle / "native-output.txt").read_text())
    native_host = parse_outputs((host / "build/native-output.txt").read_text())
    observed = {}
    host_observed = {}
    for role, i in dict(score=6, probability=7, context=8, delta=9).items():
        for base, store, dirname in (
            (bundle, observed, "native-"),
            (host, host_observed, "observe-"),
        ):
            directory = base / (
                dirname + role + ("-observation" if base == bundle else "")
            )
            report = read(directory / "observation.json")
            assert report["passed"] and report["node"] == m["nodes"][i]["id"]
            assert report["source_sha256"] == sha(bundle / "source.cpp")
            assert report["bundle_manifest_sha256"] == sha(
                directory / "input-manifest.json"
            )
            before = read(directory / "input-manifest.json")
            after = read(
                (bundle if base == bundle else host / "build") / "manifest.json"
            )
            assert before["implementation"] == after["implementation"]
            assert all(
                after["files"].get(name) == digest
                for name, digest in before["files"].items()
            )
            for name, digest in report["files"].items():
                assert sha(directory / name) == digest
            store[role] = [
                v[report["host"]]
                for v in parse_outputs((directory / "stdout.txt").read_text())
            ]
    native_checks = []
    host_checks = []
    device_checks = []
    b, p, nt, st = s["B"], s["P"], s["Nt"], s["St"]
    for e, (batch, out, hostout, d) in enumerate(
        zip(bs, native, native_host, result["diagnostics"])
    ):
        native_checks.append(
            check(b, s["N"], s["S"], batch, out, {k: v[e] for k, v in observed.items()})
        )
        host_checks.append(
            check(
                b,
                s["N"],
                s["S"],
                batch,
                hostout,
                {k: v[e] for k, v in host_observed.items()},
            )
        )
        half = lambda k: np.asarray(d[k], np.uint16).view(np.float16).astype(float)
        y = (
            lambda k: half(k)[:, 0]
            .reshape(p, b, nt)
            .transpose(1, 0, 2)
            .reshape(b, s["N"])
        )
        feature_x = (
            lambda k: half(k)[0].reshape(p, b, nt).transpose(1, 0, 2).reshape(b, s["N"])
        )
        sequence_y = (
            lambda k: half(k)[:, 0]
            .reshape(p, b, st)
            .transpose(1, 0, 2)
            .reshape(b, s["S"])
        )
        device_checks.append(
            check(
                b,
                s["N"],
                s["S"],
                batch,
                result["cases"][e],
                dict(
                    score=sequence_y("score"),
                    probability=sequence_y("probability"),
                    context=feature_x("context"),
                    delta=y("delta"),
                ),
            )
        )
    assert len(native_checks) == len(host_checks) == len(device_checks) == 8
    case = dict(
        key=item["project"] + "/" + item["kernel"],
        artifact=str(bundle.relative_to(ROOT)),
        passed=True,
        level="sdk_simulator",
        audit=audit,
        native_application_source="native-output.txt: actual C++ stdout plus four frozen intermediate observers",
        native_application_checks=native_checks,
        sdk_host_native_application_checks=host_checks,
        device_application_checks=device_checks,
        linked_static_memory=mem,
        linked_static_memory_sha256=sha(memory),
        qualification_sha256=sha(bundle / "qualification.json"),
        results_sha256=sha(bundle / "results.json"),
        source_comparison=str(comparison.relative_to(ROOT)),
        performance=dict(
            min_ratio=min(v["ratio"] for v in c["checks"]),
            max_ratio=max(v["ratio"] for v in c["checks"]),
            scope=c["scope"],
        ),
    )
    case["numerical_validation"] = numerical_summary(case)
    assert case["numerical_validation"]["fixed_accuracy_passed"]
    item.update(
        status="source_ready",
        contract="Supplied shared-cache attention with half DSR blocks, f32 local merge and SDK f32 independent-plane collectives, stable SDK softmax and output residual. No append, mask, query generation or head/GQA selection. Eight-call simulator qualification and original vecmat local-compute control; not full Decode or hardware.",
        source_comparison=str(comparison.relative_to(ROOT)),
        qualification="Eight actual SDK calls; all five original-input native/device stage gates, four actual native observers on two hosts, all-PE cache/intermediate witnesses, linked ELF and >=40 frozen mutation rejections. Local-compute control performance only.",
    )
    controls = [
        comparison,
        memory,
        mutations,
        host / "review.json",
        host / "provenance.json",
        ROOT / "cache_attention_fixtures.py",
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
    for n in ("bundle", "comparison", "memory", "mutations", "host"):
        p.add_argument(n, type=Path)
    register(**vars(p.parse_args()))
