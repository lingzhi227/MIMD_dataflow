"""Actual Clang/native build for reordered declarations; identical qualified CSL."""

import argparse, datetime, json, sys, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "toolchain"), str(ROOT)]
from compile import build
from native_transport import parse_outputs

p = argparse.ArgumentParser()
p.add_argument("qualified", type=Path)
a = p.parse_args()
q = a.qualified
assert json.loads((q / "qualification.json").read_text())["success"]
root = (
    ROOT
    / "evidence"
    / (
        "resident-input-order-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
)
root.mkdir()
s = (q / "source.cpp").read_text()
lines = s.splitlines(True)
line = next(v for v in lines if '>("q")' in v)
lines.remove(line)
lines.insert(next(i for i, v in enumerate(lines) if '>("x")' in v), line)
source = root / "hls.cpp"
source.write_text("".join(lines))
m = json.loads((q / "semantic.json").read_text())
b = json.loads((q / "batches.json").read_text())
schedule = json.loads((q / "schedule.json").read_text())
d = root / "build"
build(
    source,
    d,
    epochs=m["epochs"],
    bound=1,
    batches=b,
    instrumentation=schedule["instrumentation"],
    sdk_options=json.loads((q / "runtime-options.json").read_text()),
)
assert json.loads((d / "schedule.json").read_text()) == schedule
for name in ("layout.csl", "pe.csl", "inference_comm.csl", "inference_routes.csl"):
    assert (d / name).read_bytes() == (q / name).read_bytes()
assert parse_outputs((d / "native-output.txt").read_text()) == parse_outputs(
    (q / "native-output.txt").read_text()
)
assert (d / "WaferLLM-LICENSE.txt").read_bytes() == (
    ROOT / "projects/waferllm/upstream/LICENSE"
).read_bytes()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(root / "comparison.json").write_text(
    json.dumps(
        dict(
            passed=True,
            new_native_execution=True,
            new_sdk_execution=False,
            scope="Independent input declarations reordered in actual source: Clang/native pass, same plan, exactly identical four CSL files to qualified bundle; no new SDK execution claimed.",
            qualified_bundle=str(q),
            qualified_manifest_sha256=sha(q / "manifest.json"),
            native_manifest_sha256=sha(d / "manifest.json"),
            license_preserved=True,
        ),
        indent=2,
    )
    + "\n"
)
print(root.relative_to(ROOT), flush=True)
