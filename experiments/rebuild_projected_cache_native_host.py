"""Rebuild all native graph observations on SDK host using frozen inputs and compiler."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import hashlib, importlib.util, json, shutil, sys
from pathlib import Path

bundle, root = [Path(p).resolve() for p in sys.argv[1:]]
assert not root.exists()
root.mkdir()
sys.path.insert(0, str(bundle / "implementation"))
from integrity import verify_bundle
from compile import build
from native_observers import observe
from native_transport import parse_outputs

verify_bundle(bundle)
s, m, bs = [
    json.loads((bundle / n).read_text())
    for n in ("schedule.json", "semantic.json", "batches.json")
]
spec = importlib.util.spec_from_file_location(
    "frozen_application", bundle / "application-reference.py"
)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
p = build(
    bundle / "source.cpp",
    root / "build",
    epochs=m["epochs"],
    bound=m["input_bound"],
    batches=bs,
    instrumentation=s["instrumentation"],
    sdk_options=json.loads((bundle / "runtime-options.json").read_text()),
)
csl = sorted(p.glob("*.csl"))
assert len(csl) == len(list(bundle.glob("*.csl"))) == 9
for f in csl:
    assert f.read_bytes() == (bundle / f.name).read_bytes(), f.name
observations = []
observed = {}
for k, i in dict(
    normalized=10,
    query=11,
    key_projection=12,
    rotated_query=14,
    score=17,
    probability=18,
    context=19,
    delta=20,
).items():
    observed[k], report = observe(p, m["nodes"][i]["id"], root / ("observe-" + k))
    observations.append(report)
outputs = parse_outputs((p / "native-output.txt").read_text())
assert len(outputs) == len(bs) == 8
checks = [
    ref.check(s["B"], s["N"], s["S"], b, o, {k: v[e] for k, v in observed.items()})
    for e, (b, o) in enumerate(zip(bs, outputs))
]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copyfile(__file__, root / "driver.py")
(root / "review.json").write_text(
    json.dumps(
        dict(
            passed=True,
            checks=checks,
            observations=observations,
            csl_identical=True,
            csl_files=[p.name for p in csl],
            prior_manifest_sha256=sha(bundle / "manifest.json"),
            reference_sha256=sha(bundle / "application-reference.py"),
        ),
        indent=2,
    )
    + "\n"
)
(root / "provenance.json").write_text(
    json.dumps(
        dict(
            files={
                str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()
            }
        ),
        indent=2,
    )
    + "\n"
)
print("PASS", root)
