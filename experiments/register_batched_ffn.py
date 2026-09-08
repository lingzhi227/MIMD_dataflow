"""Admit one full SDK FFN only after independent stage and source-control gates."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from native_transport import parse_outputs
from batched_ffn_fixtures import batches, check
from run_profiles import numerical_summary

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def register(bundle, comparison, memory, mutations, host):
    bundle, comparison, memory, mutations, host = [
        Path(p).resolve() for p in (bundle, comparison, memory, mutations, host)
    ]
    item = read(bundle.parent / "PORT.json")
    catalog = read(ROOT / "benchmarks/catalog.json")
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
    assert s["profile"] == "mesh_batched_feed_forward.v1" and (
        s["B"],
        s["N"],
        s["F"],
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
        ROOT / "experiments/audit_batched_ffn_mutations.py"
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
    for role, i in dict(
        normalized=5, up=6, gate=7, activation=8, hidden=9, delta=10
    ).items():
        for base, store, dirname in (
            (bundle, observed, "native-observe-"),
            (host, host_observed, "observe-"),
        ):
            directory = base / (dirname + role)
            report = read(directory / "observation.json")
            assert report["passed"] and report["node"] == m["nodes"][i]["id"]
            assert report["source_sha256"] == sha(bundle / "source.cpp")
            input_manifest = (
                bundle if base == bundle else host / "build"
            ) / "manifest.json"
            assert report["bundle_manifest_sha256"] == sha(input_manifest)
            for name, digest in report["files"].items():
                assert sha(directory / name) == digest
            store[role] = [
                v[report["host"]]
                for v in parse_outputs((directory / "stdout.txt").read_text())
            ]
    native_checks = []
    host_checks = []
    device_checks = []
    b, p, nt, ft = s["B"], s["P"], s["Nt"], s["Ft"]
    for e, (batch, out, hostout, d) in enumerate(
        zip(bs, native, native_host, result["diagnostics"])
    ):
        native_checks.append(
            check(b, s["N"], s["F"], batch, out, {k: v[e] for k, v in observed.items()})
        )
        host_checks.append(
            check(
                b,
                s["N"],
                s["F"],
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
        x = lambda k: half(k)[0].reshape(p, b, ft).transpose(1, 0, 2).reshape(b, s["F"])
        packed = half("projections")[0]
        up = packed[:, : b * ft].reshape(p, b, ft).transpose(1, 0, 2).reshape(b, s["F"])
        gate = (
            packed[:, b * ft :].reshape(p, b, ft).transpose(1, 0, 2).reshape(b, s["F"])
        )
        device_checks.append(
            check(
                b,
                s["N"],
                s["F"],
                batch,
                result["cases"][e],
                dict(
                    normalized=y("normalized"),
                    up=up,
                    gate=gate,
                    activation=x("activation"),
                    hidden=x("hidden"),
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
        native_application_source="native-output.txt: actual C++ stdout plus six frozen intermediate observers",
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
        contract="Resident full FFN with half local DSR arithmetic and explicit SDK f32 reduce/broadcast on independent X/Y planes. Stable SDK half SiLU repairs the source approximation. Eight-call SDK qualification; original vecmat compute-control comparison only, not full Decode/cache/hardware coverage.",
        source_comparison=str(comparison.relative_to(ROOT)),
        qualification="Eight actual SDK calls; all seven native/device original-input stage gates, six actual native observers on two hosts, all-PE raw witnesses, linked ELF, >=40 frozen mutation rejections; original vecmat compute-control performance scope only. No unmodified Decode or hardware throughput claim.",
    )
    controls = [
        comparison,
        memory,
        mutations,
        host / "review.json",
        host / "provenance.json",
        ROOT / "tests/support/batched_ffn_fixtures.py",
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
    (ROOT / "benchmarks/catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(out, len(catalog))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for n in ("bundle", "comparison", "memory", "mutations", "host"):
        p.add_argument(n, type=Path)
    register(**vars(p.parse_args()))
