"""Rebuild the frozen public HLS and thirteen native branches on the SDK host."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, hashlib, json, shutil, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
bundle = a.bundle.resolve()
output = a.output.resolve()
assert not output.exists()
output.mkdir(parents=True)
sys.path[:0] = [str(bundle / "implementation"), str(bundle)]
from integrity import verify_bundle
from compile import build
from native_observers import observe
from native_transport import parse_outputs
from input_attention_fixtures import check

verify_bundle(bundle)
bs = json.loads((bundle / "batches.json").read_text())
new = build(
    bundle / "source.cpp",
    output / "build",
    epochs=8,
    bound=2,
    batches=bs,
    instrumentation="counters",
)
m = json.loads((new / "semantic.json").read_text())
s = json.loads((new / "schedule.json").read_text())
for f in new.glob("*.csl"):
    assert f.read_bytes() == (bundle / f.name).read_bytes(), f.name
indices = dict(
    input_normalized=11,
    q_raw=12,
    k_raw=13,
    v_raw=14,
    q=15,
    k=16,
    score=18,
    probability=19,
    attention=20,
    projection=21,
    z=22,
    normalized_z=23,
    delta=28,
)
values = {
    name: observe(new, m["nodes"][i]["id"], output / ("observed-" + name))[0]
    for name, i in indices.items()
}
original = parse_outputs((new / "native-output.txt").read_text())
checks = []
for e, (b, o) in enumerate(zip(bs, original)):
    obs = {name: v[e] for name, v in values.items()}
    obs["v"] = obs["v_raw"]
    checks.append(check(64, 64, 256, s["epsilon"], s["scale"], b, o, obs))
shutil.copyfile(__file__, output / "driver.py")
result = dict(
    passed=True,
    scope=__doc__,
    epochs=8,
    observed_branches=13,
    generated_csl_identical=True,
    checks=checks,
    prior_manifest_sha256=hashlib.sha256(
        (bundle / "manifest.json").read_bytes()
    ).hexdigest(),
)
(output / "review.json").write_text(json.dumps(result, indent=2) + "\n")
(output / "provenance.json").write_text(
    json.dumps(
        dict(
            files={
                str(f.relative_to(output)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in output.rglob("*")
                if f.is_file()
            }
        ),
        indent=2,
    )
    + "\n"
)
print("PASS", output, flush=True)
