"""Exercise current completed-call watchdog using preserved SDK observations.

This is middleware regression evidence, never qualification of a partial run.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from incremental_audit import make_checker


def review(bundle, observations, output):
    bundle, observations, output = map(Path, (bundle, observations, output))
    assert not output.exists()
    output.mkdir()
    for name in ("schedule.json", "semantic.json", "batches.json"):
        shutil.copyfile(bundle / name, output / name)
    snapshot = json.loads(observations.read_text())
    assert snapshot["cases"] and not snapshot["success"]

    def write(value):
        temporary = output / "results.json.tmp"
        temporary.write_text(json.dumps(value))
        temporary.replace(output / "results.json")

    shutil.copyfile(__file__, output / "driver.py")
    shutil.copyfile(
        ROOT / "lib/Debug/incremental_audit.py", output / "incremental_audit.py"
    )
    write(snapshot)
    shutil.copyfile(output / "results.json", output / "valid-prefix-results.json")
    checker = make_checker(output)
    assert checker is not None
    checker()
    report_path = output / "completed-audit.json"
    passed = json.loads(report_path.read_text())
    assert passed["passed"] and not passed["audit"]["full_run_passed"]
    stamp = report_path.stat().st_mtime_ns
    checker()
    assert report_path.stat().st_mtime_ns == stamp
    write(snapshot)  # Atomic replacement with the same payload.
    checker()
    assert report_path.stat().st_mtime_ns == stamp
    shutil.copyfile(report_path, output / "valid-prefix-audit.json")
    wrong = copy.deepcopy(snapshot)
    wrong["cases"][0]["branch0"][0] += 1.0
    write(wrong)
    shutil.copyfile(output / "results.json", output / "corrupted-prefix-results.json")
    try:
        checker()
    except Exception as error:
        numerical_error = repr(error)
    else:
        raise AssertionError("corrupted completed result was accepted")
    failed = json.loads(report_path.read_text())
    assert (
        not failed["passed"]
        and failed["results_sha256"]
        == hashlib.sha256((output / "results.json").read_bytes()).hexdigest()
    )
    shutil.copyfile(report_path, output / "corrupted-prefix-audit.json")
    regressed = copy.deepcopy(snapshot)
    regressed["cases"] = []
    write(regressed)
    try:
        checker()
    except ValueError as error:
        assert "regressed" in str(error)
    else:
        raise AssertionError("completed count regression was accepted")
    summary = dict(
        passed=True,
        scope="Current middleware regression on preserved partial SDK observations; no application qualification",
        valid_prefix_calls=len(snapshot["cases"]),
        duplicate_snapshots_skipped=True,
        numerical_failure_rejected=numerical_error,
        count_regression_rejected=True,
        source_sha256=hashlib.sha256(observations.read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(
            (ROOT / "lib/Debug/incremental_audit.py").read_bytes()
        ).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "review.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(output / "review.json")


if __name__ == "__main__":
    review(*sys.argv[1:])
