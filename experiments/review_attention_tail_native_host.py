"""Actual SDK-host C++ branch checks against a byte-identical executed CSL bundle."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, hashlib, json, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("sdk", type=Path)
p.add_argument("native", type=Path)
p.add_argument("third_party/references", type=Path)
p.add_argument("report", type=Path)
a = p.parse_args()
a.sdk = a.sdk.resolve()
a.native = a.native.resolve()
a.references = a.references.resolve()
assert not a.report.exists()
sys.path[:0] = [str(a.native / "implementation"), str(a.references)]
from integrity import verify_bundle, verify_codegen
from native_observers import observe
from application_gate import seal
from attention_tail_fixtures import check

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sm = verify_bundle(a.sdk)
nm = verify_bundle(a.native)
verify_codegen(a.sdk)
verify_codegen(a.native)
assert sm["source_sha256"] == nm["source_sha256"]
for name in ["source.cpp", "batches.json"] + [
    k for k in sm["files"] if k.endswith(".csl")
]:
    assert (a.sdk / name).read_bytes() == (a.native / name).read_bytes(), name
command = json.loads((a.native / "native-command.json").read_text())
assert command[0] == "/usr/bin/clang++-17"
front = json.loads((a.native / "00_frontend_command.json").read_text())
assert front[0] == "/usr/bin/clang++-17"
s = json.loads((a.native / "schedule.json").read_text())
m = json.loads((a.native / "semantic.json").read_text())
indices = dict(score=10, probability=11, attention=12, projection=13, delta=20)
paths = [a.native / ("native-" + name + "-observation") for name in indices]
values = [
    observe(a.native, m["nodes"][index]["id"], path)[0]
    for index, path in zip(indices.values(), paths)
]
pairs = iter(zip(*values))


def checked(b, o):
    score, probability, attention, projection, delta = next(pairs)
    return check(
        s["M"],
        s["N"],
        s["F"],
        s["epsilon"],
        s["scale"],
        b,
        o,
        attention,
        projection,
        delta,
        score=score,
        probability=probability,
    )


gate = seal(
    a.native,
    a.references / "tests/support/attention_tail_fixtures.py",
    checked,
    dict(M=s["M"], N=s["N"], F=s["F"]),
    native_observations=paths,
)
report = dict(
    passed=gate["passed"],
    new_sdk_execution=False,
    source_sdk=str(a.sdk),
    native_bundle=str(a.native),
    generated_csl_byte_identical=True,
    native_compiler=command[0],
    actual_native_checks=gate["checks"],
    sdk_manifest_sha256=sha(a.sdk / "manifest.json"),
    native_manifest_sha256=sha(a.native / "manifest.json"),
    driver_sha256=sha(Path(__file__)),
    reference_hashes={v.name: sha(v) for v in a.references.glob("*.py")},
    scope="Fresh actual C++ compilation and five observed native branches on SDK host, frozen frontend/native runtime, original eight inputs, independent fsum/sqrt/exp. CSL identical to source SDK bundle; this host check is not another SDK execution.",
)
a.report.write_text(json.dumps(report, indent=2) + "\n")
print(a.report, flush=True)
