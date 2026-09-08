"""Exercise the standard runner's native/target gates in a fresh local bundle."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import datetime
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib"), str(ROOT / "experiments")]
from compile import build
from composed_ffn_fixtures import batches
from composed_ffn_gate import seal_native, seal_target
from integrity import verify_bundle, verify_codegen
from run_profiles import numerical_summary


def main():
    root = (
        ROOT
        / "validation/evidence"
        / (
            "composed-standard-gates-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    shutil.copy2(__file__, root / "driver.py")
    try:
        bundle = root / "bundle"
        build(
            ROOT / "benchmarks/inference/waferllm/projected_cache_ffn_3x256x512x512_16x16/hls.cpp",
            bundle,
            epochs=8,
            bound=2,
            batches=batches(3, 256, 512, 512),
            instrumentation="sampled",
            sdk_options=dict(suppress_trace=True, num_threads=8, dump_core=False),
        )
        native = seal_native(bundle)
        target = seal_target(bundle)
        summary = numerical_summary(dict(native_application_checks=native["checks"]))
        assert (
            native["passed"]
            and target["passed"]
            and summary["native_fixed_accuracy_passed"]
        )
        verify_bundle(bundle)
        verify_codegen(bundle)
        files = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }
        (root / "report.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    native=native["checks"],
                    target=target["checks"],
                    summary=summary,
                    files=files,
                    scope="Standard runner's fresh native observers, independent original-input stage gates and target preflight. No new SDK execution.",
                ),
                indent=2,
            )
            + "\n"
        )
        print(root, flush=True)
    except BaseException:
        (root / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
