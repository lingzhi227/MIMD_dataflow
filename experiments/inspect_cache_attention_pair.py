"""Compare saved HLS/source calls without admitting an unfinished SDK batch."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
p.add_argument("source", type=Path)
p.add_argument("report", type=Path)
a = p.parse_args()
assert not a.report.exists()
root = a.bundle.resolve()
source = a.source.resolve()
sys.path.insert(0, str(root / "implementation"))
from integrity import verify_bundle, verify_codegen
from cache_attention_reference import audit_cases
from mesh_cache_attention_sdk import packed

verify_bundle(root)
verify_codegen(root)
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
s, m, batches, h = [
    read(root / n)
    for n in ("schedule.json", "semantic.json", "batches.json", "results.json")
]
review = audit_cases(s, m, batches, h, require_complete=False)
prov = read(source / "provenance.json")
for name, digest in prov["files"].items():
    assert sha(source / name) == digest
assert prov["hls_manifest_sha256"] == sha(root / "manifest.json")
r = read(source / "results.json")
inputs = read(source / "inputs.json")
count = min(len(h["cases"]), len(r["cases"]))
assert count > 0
rows = []
for e in range(count):
    for name, v in packed(s, m, batches[e]).items():
        np.testing.assert_array_equal(v, inputs[e][name])
    left, right = h["diagnostics"][e], r["cases"][e]
    assert set(left) == set(right)
    for name, v in left.items():
        if name not in ("timing", "queues"):
            np.testing.assert_array_equal(v, right[name], err_msg=f"{e} {name}")
    assert np.all((np.asarray(right["queues"]) & 60) == 60)

    def cycles(d):
        t = np.asarray(d["timing"], np.uint64)
        a = t[:, :, 0] + (t[:, :, 1] << 16) + (t[:, :, 2] << 32)
        b = t[:, :, 3] + (t[:, :, 4] << 16) + (t[:, :, 5] << 32)
        v = (b - a) & ((1 << 48) - 1)
        assert np.all((v > 0) & (v < 2**32))
        return int(v.max())

    lc, rc = cycles(left), cycles(right)
    rows.append(dict(epoch=e, hls_cycles=lc, source_cycles=rc, ratio=lc / rc))
a.report.write_text(
    json.dumps(
        dict(
            passed=True,
            full_qualification=False,
            paired_completed_calls=count,
            hls_completed_audit=review,
            cases=rows,
            scope="Saved call snapshots only, no final execution/SIF completion admission",
            files={
                str(p): sha(p)
                for p in (
                    root / "manifest.json",
                    root / "results.json",
                    source / "provenance.json",
                    source / "results.json",
                    Path(__file__),
                )
            },
        ),
        indent=2,
    )
    + "\n"
)
print(count, rows)
