"""Register bounded FFT profiles only after preserved SDK and source controls pass."""

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
import subprocess
import sys
from pathlib import Path

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from fft_fixtures import check
from native_transport import parse_outputs
from run_profiles import numerical_summary

PROFILES = {
    "fft3d_16_4x4_forward": "191005684402",
    "fft3d_16_4x4_inverse_backward": "191338037614",
    "fft3d_16_4x4_forward_ortho": "191339306224",
    "fft3d_16_4x4_inverse_ortho": "191340556542",
    "fft3d_16_4x4_forward_forward": "191341856403",
    "fft3d_16_4x4_inverse_forward": "191343132301",
    "fft3d_32_8x8_forward_backward": "192937261157",
    "fft3d_32_8x8_forward_backward_transposed": "195708248348",
    "fft3d_32_8x8_inverse_backward": "201357164868",
    "fft3d_64_16x16_forward_backward": "192439114535",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("native64_control", type=Path)
    args = parser.parse_args()
    controls = [
        ROOT / "validation/evidence/fft-native-20260906T191255216720Z",
        ROOT / "validation/evidence/fft-native-20260906T203739488954Z",
        args.native64_control.resolve(),
    ]
    for size, control in zip((16, 32, 64), controls):
        assert read(control / "comparison.json")["passed"]
        assert read(control / "schedule.json")["N"] == size
        assert read(control / "results.json")["success"]
    for evidence in (
        "fft32-layout-comparison.json",
        "fft32-instrumentation-comparison.json",
        "fft16-device-roundtrip-bit-review.json",
        "fft32-device-roundtrip-result.json",
    ):
        assert read(ROOT / "validation/evidence" / evidence)["passed"], evidence
    catalog = read(ROOT / "benchmarks/catalog.json")
    assert not any(
        p["kernel"] in PROFILES and p["project"] == "sdk_examples" for p in catalog
    ), "Profiles already registered; do not duplicate or overwrite history"
    cases, additions = [], []
    for name, stamp in PROFILES.items():
        folder = ROOT / "benchmarks/sdk_examples" / name
        bundle = folder / ("run-20260906T" + stamp + "Z")
        assert (folder / "hls.cpp").read_bytes() == (bundle / "source.cpp").read_bytes()
        q = read(bundle / "qualification.json")
        assert q["success"]
        code = 'import sys,json;from pathlib import Path;p=Path(sys.argv[1]).resolve();sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
        audit = json.loads(
            subprocess.check_output(
                [sys.executable, "-c", code, str(bundle)], text=True
            )
        )
        assert audit["passed"]
        s = read(bundle / "schedule.json")
        t = s["transform"]
        batches = read(bundle / "batches.json")
        native = parse_outputs((bundle / "native-output.txt").read_text())
        device = read(bundle / "results.json")["cases"]
        assert len(native) == len(device) == len(batches) == s["epochs"]
        checks = lambda values: [
            check(s["N"], t["direction"], t["norm"], b, y)
            for b, y in zip(batches, values)
        ]
        case = dict(
            key="sdk_examples/" + name,
            artifact=str(bundle.relative_to(ROOT)),
            passed=True,
            level="sdk_simulator",
            audit=audit,
            native_application_checks=checks(native),
            native_application_source="native-output.txt: actual C++ executable stdout, read-only independent recheck",
            device_application_checks=checks(device),
            schedule_phase_count=len(s["stages"]),
            qualification_sha256=sha(bundle / "qualification.json"),
            results_sha256=sha(bundle / "results.json"),
        )
        case["numerical_validation"] = numerical_summary(case)
        cases.append(case)
        item = read(folder / "PORT.json")
        item.update(
            fixture=f"distributed_fft:{s['N']}:{t['direction']}:{t['norm']}",
            status="source_ready",
            contract=f"SDK-native C2C {s['N']}³ {t['direction']}/{t['norm']} on{s['rows']}x{s['cols']}PEs; {s['epochs']}warm SDK calls, {s.get('result_layout', 'input_layout')} ownership, {len(s['stages'])}stages. Native stdout/directDFT and full device normwise/peak-scaled checks; per-component accuracy not implied. Source controls cover forward/backward-norm at16/32/64; other modes have measured local intervals, not individual source-overhead controls. Not full wsFFT/SlideFFT reproduction or hardware performance.",
        )
        additions.append((folder, item))
    # Finish every check before modifying the catalog or per-profile metadata.
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    index = ROOT / "validation/evidence" / ("qualification-" + stamp + ".json")
    assert not index.exists()
    index.write_text(
        json.dumps(
            dict(
                kind="qualification_index_of_existing_sdk_runs",
                sdk=True,
                success=True,
                new_sdk_execution=False,
                source_controls=[
                    dict(
                        path=str(p.relative_to(ROOT)),
                        comparison_sha256=sha(p / "comparison.json"),
                    )
                    for p in controls
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
    print(index.relative_to(ROOT), len(additions), "bounded profiles registered")


if __name__ == "__main__":
    main()
