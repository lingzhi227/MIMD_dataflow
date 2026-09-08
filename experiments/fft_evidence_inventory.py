"""Re-audit completed FFT bundles with frozen implementations and actual C++ output."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from fft_fixtures import check
from native_transport import parse_outputs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    rows = []
    for path in sorted(
        (ROOT / "benchmarks/sdk_examples").glob("fft3d*/run-*/qualification.json")
    ):
        qualification = json.loads(path.read_text())
        if not qualification["success"]:
            continue
        root = path.parent
        code = 'import sys,json;from pathlib import Path;p=Path(sys.argv[1]).resolve();sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
        audit = json.loads(
            subprocess.check_output([sys.executable, "-c", code, str(root)], text=True)
        )
        assert audit["passed"]
        schedule = json.loads((root / "schedule.json").read_text())
        batches = json.loads((root / "batches.json").read_text())
        native = parse_outputs((root / "native-output.txt").read_text())
        assert len(native) == len(batches) == schedule["epochs"]
        transform = schedule["transform"]
        native_checks = [
            check(schedule["N"], transform["direction"], transform["norm"], b, y)
            for b, y in zip(batches, native)
        ]
        rows.append(
            dict(
                bundle=str(root.relative_to(ROOT)),
                hashes={
                    name: sha(root / name)
                    for name in (
                        "qualification.json",
                        "results.json",
                        "native-output.txt",
                        "schedule.json",
                    )
                },
                transform=transform,
                mesh=[schedule["rows"], schedule["cols"]],
                result_layout=schedule.get("result_layout", "input_layout"),
                phase_count=len(schedule["stages"]),
                epochs=schedule["epochs"],
                internal_float_observations=audit.get("internal_float_observations", 0),
                max_local_cycles=max(
                    case["max_local_cycles"] for case in audit["cases"]
                ),
                frozen_device_audit_passed=True,
                native_stdout_application_checks=native_checks,
            )
        )
    assert rows
    args.output.write_text(
        json.dumps(
            dict(
                passed=True,
                new_native_execution=False,
                new_sdk_execution=False,
                tool_hashes={
                    str(p.relative_to(ROOT)): sha(p)
                    for p in (
                        Path(__file__).resolve(),
                        ROOT / "tests/support/fft_fixtures.py",
                        ROOT / "lib/Runtime/native_transport.py",
                        ROOT / "lib/Numerics/float32.py",
                    )
                },
                scope="Read-only re-audit of completed runs, not unique applications or a new execution. Explicit stage counts supersede historical generic free-text stage wording. Performance controls are separate artifacts; simulator local cycles do not establish hardware latency.",
                runs=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    print(len(rows), "completed runs re-audited", args.output)


if __name__ == "__main__":
    main()
