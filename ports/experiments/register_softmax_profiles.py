"""Register executed distributed softmax profiles after independent native/device review."""

import argparse, datetime, hashlib, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from softmax_fixtures import check, batches as fixtures
from native_transport import parse_outputs
from run_ports import numerical_summary


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundles", type=Path, nargs=4)
    a = p.parse_args()
    required = {
        "softmax_64x128_8x8",
        "softmax_64x128_8x8_map",
        "softmax_128x1024_8x8",
        "softmax_128x1024_8x8_map",
    }
    assert {b.parent.name for b in a.bundles} == required
    controls = [
        ROOT / "evidence" / n
        for n in (
            "half-exp-production-model-review.json",
            "softmax64x64-source-comparison.json",
            "softmax64x128-map-comparison.json",
            "softmax128x1024-map-comparison.json",
        )
    ]
    for path in controls:
        assert read(path)["passed"]
    catalog = read(ROOT / "catalog.json")
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
            fixture=f"distributed_softmax:{s['M']}:{s['N']}",
            status="source_ready",
            contract=f"Stable scaled row softmax {s['M']}x{s['N']} on {s['rows']}x{s['cols']} PEs, six warm SDK calls. Source-inspired max/sum chains with finite negative maximum initialization, explicit half accumulation, SDK exp, and {s.get('elementwise', 'scalar')} elementwise lowering. Native stdout and device independent standard-math checks; exact target half stages/results. Matched scalar/map simulator interval controls; original-source overhead control only 64x64. No hardware or full-inference claim.",
        )
        additions.append((bundle.parent, item))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = ROOT / "evidence" / ("qualification-" + stamp + ".json")
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
    (ROOT / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(path.relative_to(ROOT), len(additions))


if __name__ == "__main__":
    main()
