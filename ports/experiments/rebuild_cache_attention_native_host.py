"""SDK-host native cache attention rebuild using only the frozen compiler and input bundle."""

import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np

bundle, root = [Path(p).resolve() for p in sys.argv[1:]]
assert not root.exists()
root.mkdir()
sys.path.insert(0, str(bundle / "implementation"))
from integrity import verify_bundle
from compile import build
from native_observers import observe
from native_transport import parse_outputs
from mesh_cache_attention_sdk import values
from cache_attention_reference import original_math, accuracy

verify_bundle(bundle)
s, m, bs = [
    json.loads((bundle / n).read_text())
    for n in ("schedule.json", "semantic.json", "batches.json")
]
p = build(
    bundle / "source.cpp",
    root / "build",
    epochs=m["epochs"],
    bound=m["input_bound"],
    batches=bs,
    instrumentation=s["instrumentation"],
    sdk_options=json.loads((bundle / "runtime-options.json").read_text()),
)
for f in p.glob("*.csl"):
    assert f.read_bytes() == (bundle / f.name).read_bytes(), f.name
expected = [original_math(values(m, b), s["scale"]) for b in bs]
checks = [
    dict(
        result=accuracy(
            np.array(v[s["output_port"]]).reshape(s["B"], s["N"]), e["result"]
        )
    )
    for v, e in zip(parse_outputs((p / "native-output.txt").read_text()), expected)
]
observations = []
for i, k in ((6, "score"), (7, "probability"), (8, "context"), (9, "delta")):
    node = m["nodes"][i]
    vals, report = observe(p, node["id"], root / ("observe-" + k))
    observations.append(report)
    for c, v, e in zip(checks, vals, expected):
        c[k] = accuracy(np.array(v).reshape(node["shape"]), e[k])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copyfile(__file__, root / "driver.py")
(root / "review.json").write_text(
    json.dumps(
        dict(
            passed=True,
            checks=checks,
            observations=observations,
            csl_identical=True,
            prior_manifest_sha256=sha(bundle / "manifest.json"),
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
