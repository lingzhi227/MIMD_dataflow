"""Prove updated authoring preserves the executing CSL and original test inputs."""

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
from pathlib import Path

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from compile import build
from native_transport import parse_outputs
from composed_ffn_fixtures import batches, check


def main():
    project = ROOT / "benchmarks/inference/waferllm/projected_cache_ffn_3x256x512x512_16x16"
    old = project / "run-20260908T030001167477Z"
    dest = (
        ROOT
        / "validation/evidence"
        / (
            "composed-authoring-review-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    dest.mkdir()
    shutil.copy2(__file__, dest / "driver.py")
    for name in (
        "tests/support/composed_ffn_fixtures.py",
        "tests/support/projected_cache_fixtures.py",
        "tests/support/cache_attention_fixtures.py",
    ):
        shutil.copy2(ROOT / name, dest / name)
    rows = batches(3, 256, 512, 512)
    assert rows == json.loads(
        (old / "batches.json").read_text()
    ), "Do not change qualification inputs"
    options = json.loads((old / "build-configuration.json").read_text())
    build(
        project / "hls.cpp",
        dest / "bundle",
        epochs=8,
        bound=2,
        batches=rows,
        instrumentation="sampled",
        sdk_options=options.get("sdk_options"),
    )
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    same = {}
    for path in old.glob("*.csl"):
        other = dest / "bundle" / path.name
        assert path.read_bytes() == other.read_bytes(), path.name
        same[path.name] = sha(path)
    for name in ("source.cpp", "native-output.txt", "schedule.json", "semantic.json"):
        assert (old / name).read_bytes() == (dest / "bundle" / name).read_bytes(), name
        same[name] = sha(old / name)
    native = parse_outputs((dest / "bundle/native-output.txt").read_text())
    remote = ROOT / "validation/evidence/composed-native-host-20260908T031354352108Z"
    observed = parse_outputs((remote / "observed-output.txt").read_text())
    checks = []
    for row, output, stages in zip(rows, native, observed):
        all_stages = {
            k.removeprefix("observe_"): v
            for k, v in stages.items()
            if k.startswith("observe_")
        }
        checks.append(check(3, 256, 512, 512, row, output, all_stages))
    report = dict(
        passed=True,
        unchanged_files=same,
        checks=checks,
        files={
            str(p.relative_to(dest)): sha(p)
            for p in dest.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        },
        old_manifest_sha256=sha(old / "manifest.json"),
        observer_sha256=sha(remote / "observed-output.txt"),
        scope="Updated explicit-precision dispatch and launch validation; identical generated CSL/schedule/semantic/native outputs and all 8 input batches. Fresh native build plus 18-stage independent stdlib original-input gates. Not a new SDK execution or retroactive modification of the executing bundle.",
    )
    (dest / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(dest, flush=True)


if __name__ == "__main__":
    main()
