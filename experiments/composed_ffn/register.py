"""Admit the composed profile only from complete, mutually bound evidence.

All mutable authoring controls are copied to a new immutable archive before
writing the qualification index. This never edits an SDK bundle or synthesizes
missing observations.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib"), str(ROOT / "experiments")]
from composed_ffn_fixtures import batches, check
from composed_ffn_gate import device_checks
from native_transport import parse_outputs
from run_profiles import numerical_summary

read = lambda path: json.loads(path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def verify_report_files(root, report):
    for name, digest in report["files"].items():
        assert sha(root / name) == digest, name


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "bundle",
        "comparison",
        "memory",
        "source_memory",
        "mutations",
        "host",
        "authoring",
        "gates",
        "regression",
    ):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    bundle = a.bundle.resolve()
    project = bundle.parent
    key = "waferllm/projected_cache_ffn_3x256x512x512_16x16"
    assert project == ROOT / "benchmarks" / key
    catalog = read(ROOT / "benchmarks/catalog.json")
    assert key not in {v["project"] + "/" + v["kernel"] for v in catalog}
    s, m, bs, result = [
        read(bundle / name)
        for name in ("schedule.json", "semantic.json", "batches.json", "results.json")
    ]
    assert s["profile"] == "mesh_projected_cache_ffn.v1"
    assert bs == batches(3, 256, 512, 512)
    assert result["success"] is True and result["launches"] == ["hls_main"] * 8
    assert len(result["cases"]) == len(result["diagnostics"]) == 8
    execution = read(bundle / "execution.json")
    assert execution["success"] is True and execution["results_sha256"] == sha(
        bundle / "results.json"
    )
    assert execution["manifest_sha256"] == sha(bundle / "manifest.json")
    assert execution["controller_sha256"] == sha(bundle / "execution-driver.py")
    assert (
        execution["sdk_sha256"]
        == "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"
    )
    code = "import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/'implementation'));from mesh_projected_cache_ffn_sdk import audit;print(json.dumps(audit(p)))"
    audit = json.loads(
        subprocess.check_output([sys.executable, "-c", code, str(bundle)], text=True)
    )
    assert audit["passed"] and audit["complete"] and audit["completed_calls"] == 8
    comparison = read(a.comparison)
    verify_report_files(ROOT, comparison)
    assert (
        comparison["passed"] and comparison["epochs"] == len(comparison["checks"]) == 8
    )
    assert comparison["files"][str((bundle / "results.json").relative_to(ROOT))] == sha(
        bundle / "results.json"
    )
    mutation = read(a.mutations)
    assert (
        mutation["passed"]
        and mutation["baseline"]["complete"]
        and mutation["baseline"]["completed_calls"] == 8
    )
    assert mutation["mutations"] == len(mutation["cases"]) >= 119
    assert len({v["name"] for v in mutation["cases"]}) == len(mutation["cases"])
    assert all(v["rejected"] is True for v in mutation["cases"])
    assert mutation["results_sha256"] == sha(bundle / "results.json")
    assert mutation["manifest_sha256"] == sha(bundle / "manifest.json")
    assert mutation["driver_sha256"] == sha(
        Path(__file__).with_name("mutations_full.py")
    )
    memory = read(a.memory)
    assert memory["elf_classes"] == len(memory["classes"]) == 9
    assert memory["max_static_high_water_bytes"] == max(
        v["static_high_water_bytes"] for v in memory["classes"]
    )
    assert (
        memory["max_static_high_water_bytes"]
        <= sum(s["memory_per_pe"].values())
        <= 49152
    )
    assert {v["elf"] for v in memory["classes"]} == {
        p.name for p in (bundle / "out/bin").glob("*.elf")
    }
    for v in memory["classes"]:
        assert sha(bundle / "out/bin" / v["elf"]) == v["sha256"]
        assert v["static_high_water_bytes"] == max(
            w["start"] + w["size"] for w in v["sections"]
        )
    sources = [
        ROOT / name for name in comparison["files"] if name.endswith("/provenance.json")
    ]
    assert len(sources) == 1
    source = sources[0].parent
    source_memory = read(a.source_memory)
    assert source_memory["elf_classes"] == len(source_memory["classes"]) == 9
    assert {v["elf"] for v in source_memory["classes"]} == {
        p.name for p in (source / "out/bin").glob("*.elf")
    }
    assert source_memory["max_static_high_water_bytes"] == max(
        v["static_high_water_bytes"] for v in source_memory["classes"]
    )
    for v in source_memory["classes"]:
        assert sha(source / "out/bin" / v["elf"]) == v["sha256"]
        assert v["static_high_water_bytes"] == max(
            w["start"] + w["size"] for w in v["sections"]
        )
        assert v["static_high_water_bytes"] <= 49152
    host, authoring, gates = [
        read(path / "report.json") for path in (a.host, a.authoring, a.gates)
    ]
    for path, report in ((a.host, host), (a.authoring, authoring), (a.gates, gates)):
        assert report["passed"]
        verify_report_files(path, report)
    assert host["native_cross_host_exact"] and host["epochs"] == 8
    assert authoring["old_manifest_sha256"] == sha(bundle / "manifest.json")
    assert sha(a.host / "original-manifest.json") == sha(bundle / "manifest.json")
    assert (a.host / "source.cpp").read_bytes() == (bundle / "source.cpp").read_bytes()
    assert (a.host / "native-input.txt").read_bytes() == (
        bundle / "native-input.txt"
    ).read_bytes()
    assert (a.host / "source-output.txt").read_bytes() == (
        bundle / "native-output.txt"
    ).read_bytes()
    for other in (a.authoring / "bundle", a.gates / "bundle"):
        for path in bundle.glob("*.csl"):
            assert path.read_bytes() == (other / path.name).read_bytes(), path.name
        assert read(other / "batches.json") == bs
        assert (other / "native-output.txt").read_bytes() == (
            bundle / "native-output.txt"
        ).read_bytes()
    observed = parse_outputs((a.host / "observed-output.txt").read_text())
    native = parse_outputs((bundle / "native-output.txt").read_text())
    native_checks = [
        check(
            3,
            256,
            512,
            512,
            b,
            o,
            {
                k.removeprefix("observe_"): v
                for k, v in obs.items()
                if k.startswith("observe_")
            },
        )
        for b, o, obs in zip(bs, native, observed)
    ]
    device = device_checks(bundle)
    assert (
        len(native_checks)
        == len(device)
        == len(gates["native"])
        == len(gates["target"])
        == 8
    )
    assert (
        "Ran 332 tests" in a.regression.read_text()
        and a.regression.read_text().rstrip().endswith("OK")
    )
    mapping = read(project / "SOURCE-MAP.json")
    assert (
        mapping["hls_sha256"] == sha(bundle / "source.cpp") == sha(project / "hls.cpp")
    )
    upstream = ROOT / mapping["source"]
    assert mapping["source_sha256"] == sha(upstream) and len(mapping["stages"]) == 18
    lines = upstream.read_text().splitlines()
    for row in mapping["stages"]:
        body = "\n".join(lines[row["source_start_line"] - 1 : row["source_end_line"]])
        assert hashlib.sha256(body.encode()).hexdigest() == row["function_sha256"]
    item = dict(
        project="waferllm",
        kernel=project.name,
        status="source_ready",
        partitions=1,
        fixture="composed_ffn:3:256:512:512",
        instrumentation="sampled",
        source_commit=mapping["source_commit"],
        source_map="SOURCE-MAP.json",
        origins=[
            "Decode/src/decode.csl:" + v["source_function"] for v in mapping["stages"]
        ],
        contract="Resident 16x16 normalized QKV/pair/cache attention/output/residual plus mean-RMS/SwiGLU FFN, shared gamma and original Z residual. Explicit half blocks/f32 merge, shared SDK planes and one host completion. Old cache read-only; new K/V outputs; no append/head/GQA/mask. Eight simulator calls and nine-local-contraction source control, not full Decode or hardware.",
        source_comparison=str(a.comparison.resolve().relative_to(ROOT)),
    )
    case = dict(
        key=key,
        artifact=str(bundle.relative_to(ROOT)),
        passed=True,
        level="sdk_simulator",
        audit=audit,
        native_application_source="Actual original C++ stdout plus frozen eighteen-stage observed C++ on two hosts; separate standard native observer gate",
        native_application_checks=native_checks,
        device_application_checks=device,
        linked_static_memory=memory,
        source_static_memory=source_memory,
        results_sha256=sha(bundle / "results.json"),
        execution_sha256=sha(bundle / "execution.json"),
        source_comparison=item["source_comparison"],
        performance=dict(
            min_ratio=min(v["hls_over_source"] for v in comparison["checks"]),
            max_ratio=max(v["hls_over_source"] for v in comparison["checks"]),
            scope=comparison["scope"],
        ),
    )
    case["numerical_validation"] = numerical_summary(case)
    assert case["numerical_validation"]["fixed_accuracy_passed"]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = ROOT / "validation/evidence" / ("qualification-controls-" + stamp)
    archive.mkdir()
    controls = [
        a.comparison,
        a.memory,
        a.source_memory,
        a.mutations,
        a.regression,
        a.host / "report.json",
        a.authoring / "report.json",
        a.gates / "report.json",
        project / "SOURCE-MAP.json",
        ROOT / "tests/support/composed_ffn_fixtures.py",
        ROOT / "tests/support/projected_cache_fixtures.py",
        ROOT / "tests/support/cache_attention_fixtures.py",
        ROOT / "experiments/composed_ffn_gate.py",
        ROOT / "tools/run_profiles.py",
        Path(__file__),
        Path(__file__).with_name("mutations_full.py"),
        Path(__file__).with_name("compare_full.py"),
        ROOT / "lib/Numerics/projected_cache_ffn_reference.py",
        ROOT / "lib/IR/ir.py",
    ]
    records = []
    for path in controls:
        path = path.resolve()
        relative = path.relative_to(ROOT)
        dest = archive / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        records.append(
            dict(
                path=str(dest.relative_to(ROOT)),
                sha256=sha(dest),
                original_path=str(relative),
            )
        )
    report = ROOT / "validation/evidence" / ("qualification-" + stamp + ".json")
    report.write_text(
        json.dumps(
            dict(
                kind="qualification_index_of_existing_sdk_runs",
                sdk=True,
                success=True,
                new_sdk_execution=False,
                cases=[case],
                source_controls=records,
                audit_extension="Frozen 030001 numeric audit plus hash-bound supplemental launch-order/premature-success guards; not retroactively part of original SDK bundle.",
            ),
            indent=2,
        )
        + "\n"
    )
    (project / "PORT.json").write_text(json.dumps(item, indent=2) + "\n")
    catalog.append(item)
    (ROOT / "benchmarks/catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(report, len(catalog), flush=True)


if __name__ == "__main__":
    main()
