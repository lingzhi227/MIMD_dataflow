"""Register executed distributed RMS profiles after independent native/device review."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from rms_fixtures import check, batches as fixtures
from native_transport import parse_outputs
from run_profiles import numerical_summary


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundles", type=Path, nargs=3)
    a = p.parse_args()
    required = {"rmsnorm_64x128_8x8", "rmsnorm_128x1024_8x8", "rmsnorm_128x2048_8x16"}
    assert {b.parent.name for b in a.bundles} == required
    controls = [
        ROOT / "validation/evidence" / n
        for n in (
            "rms-math-exhaustive-analysis-20260906T2151Z.json",
            "rms64x128-counter-source-comparison.json",
            "rms64x128-instrumentation-comparison.json",
        )
    ]
    for path in controls:
        assert read(path)["passed"]
    catalog = read(ROOT / "benchmarks/catalog.json")
    assert not any(
        v["project"] == "waferllm" and v["kernel"] in required for v in catalog
    ), "preserve existing registration"
    cases = []
    additions = []
    for bundle in a.bundles:
        bundle = bundle.resolve()
        q = read(bundle / "qualification.json")
        assert q["success"]
        code = 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
        audit = json.loads(
            subprocess.check_output(
                [sys.executable, "-c", code, str(bundle)], text=True
            )
        )
        assert audit["passed"]
        s = read(bundle / "schedule.json")
        b = read(bundle / "batches.json")
        native = parse_outputs((bundle / "native-output.txt").read_text())
        device = read(bundle / "results.json")["cases"]
        assert (
            b == fixtures(s["M"], s["N"]) and len(native) == len(device) == len(b) == 6
        )
        checks = lambda outputs: [
            check(s["M"], s["N"], x, y) for x, y in zip(b, outputs)
        ]
        case = dict(
            key="waferllm/" + bundle.parent.name,
            artifact=str(bundle.relative_to(ROOT)),
            passed=True,
            level="sdk_simulator",
            audit=audit,
            native_application_checks=checks(native),
            native_application_source="native-output.txt: actual C++ executable stdout, independent math.fsum recheck",
            device_application_checks=checks(device),
            qualification_sha256=sha(bundle / "qualification.json"),
            results_sha256=sha(bundle / "results.json"),
        )
        case["numerical_validation"] = numerical_summary(case)
        cases.append(case)
        item = read(bundle.parent / "PORT.json")
        item.update(
            fixture=f"distributed_rms:{s['M']}:{s['N']}",
            status="source_ready",
            contract=f"Standard row RMSNorm {s['M']}x{s['N']} on{s['rows']}x{s['cols']}PEs, six warm SDK calls. Source-inspired bidirectional chain, explicit half accumulation and SDK math. Original feature-indexed scale and weight ownership corrected. Native stdout and device standard normwise checks, exact target half stages/results. Source-overhead control only64x128; larger shapes measured local intervals, no hardware/full-inference claim.",
        )
        additions.append((bundle.parent, item))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = ROOT / "validation/evidence" / ("qualification-" + stamp + ".json")
    assert not path.exists()
    path.write_text(
        json.dumps(
            dict(
                kind="qualification_index_of_existing_sdk_runs",
                sdk=True,
                success=True,
                new_sdk_execution=False,
                source_controls=[
                    dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in controls
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


if __name__ == "__main__":
    main()
