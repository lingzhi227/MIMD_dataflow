"""Compare saved HLS/source calls without admitting an unfinished SDK batch."""

import argparse, hashlib, json, sys, shutil
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
from projected_cache_reference import audit_cases
from mesh_projected_cache_sdk import packed

verify_bundle(root)
verify_codegen(root)
snapshot_dir = a.report.with_suffix(".inputs")
assert not snapshot_dir.exists()
snapshot_dir.mkdir()
snapshots = {}
for name, path in (
    ("hls-manifest.json", root / "manifest.json"),
    ("hls-results.json", root / "results.json"),
    ("source-provenance.json", source / "provenance.json"),
    ("source-results.json", source / "results.json"),
    ("driver.py", Path(__file__)),
):
    saved = snapshot_dir / name
    saved.write_bytes(path.read_bytes())
    snapshots[path] = saved
read = lambda p: json.loads(snapshots.get(p, p).read_text())
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
            files={str(p.resolve()): sha(p) for p in snapshot_dir.iterdir()},
            original_bundle=str(root),
            original_source=str(source),
            snapshot_directory=str(snapshot_dir.resolve()),
        ),
        indent=2,
    )
    + "\n"
)
print(count, rows)
