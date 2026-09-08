"""Freeze qualification source witnesses so later authoring does not invalidate history.

The original admission remains unchanged. A new index points to byte-identical
control snapshots and the same already executed cases; no new SDK run is claimed.
"""

import argparse, datetime, hashlib, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("admission", type=Path)
a = p.parse_args()
old = a.admission.resolve()
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
r = read(old)
assert r["success"] and r["sdk"] and not r["new_sdk_execution"]
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
archive = ROOT / "evidence" / ("qualification-controls-" + stamp)
archive.mkdir()
controls = []
for control in r["source_controls"]:
    relative = Path(control["path"])
    assert not relative.is_absolute() and ".." not in relative.parts
    src = ROOT / relative
    assert sha(src) == control["sha256"], relative
    dest = archive / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    assert sha(dest) == control["sha256"]
    controls.append(
        dict(
            path=str(dest.relative_to(ROOT)),
            sha256=control["sha256"],
            original_authoring_path=str(relative),
        )
    )
shutil.copyfile(__file__, archive / "archive-driver.py")
controls.append(
    dict(
        path=str((archive / "archive-driver.py").relative_to(ROOT)),
        sha256=sha(archive / "archive-driver.py"),
    )
)
new = dict(
    r,
    source_controls=controls,
    source_control_archive=dict(
        original_admission=str(old.relative_to(ROOT)),
        original_admission_sha256=sha(old),
        scope="Byte-identical source witnesses frozen for subsequent development; original report and SDK outputs are unchanged. This is an index amendment, not a new configuration or execution.",
    ),
)
report = ROOT / "evidence" / ("qualification-" + stamp + ".json")
assert not report.exists()
report.write_text(json.dumps(new, indent=2) + "\n")
for c in new["source_controls"]:
    assert sha(ROOT / c["path"]) == c["sha256"]
assert new["cases"] == r["cases"]
print(report.relative_to(ROOT))
print(archive.relative_to(ROOT))
print(len(controls), "frozen controls")
