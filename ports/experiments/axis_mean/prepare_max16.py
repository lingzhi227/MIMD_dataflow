"""Independent 16-way SDK SUM/MAX layout witness, preserving the 8-way run."""

import datetime, hashlib, json, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
base = ROOT / "evidence/sdk-axis-max-20260907T213834656626Z"
r = (
    ROOT
    / "evidence"
    / (
        "sdk-axis-max16-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
)
r.mkdir()
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for f, h in read(base / "provenance.json")["files"].items():
    assert sha(base / f) == h
assert read(base / "execution.json")["success"]
for n in (
    "sdk_axis_reduce.csl",
    "sdk_axis_max.csl",
    "runtime-options.json",
    "driver.py",
    "probe_runtime.py",
    "sdk_process.py",
):
    shutil.copyfile(base / n, r / n)
layout = (
    (base / "layout.csl")
    .read_text()
    .replace(".width=8,.height=8", ".width=16,.height=16")
    .replace("@set_rectangle(8,8)", "@set_rectangle(16,16)")
    .replace("@range(u16,8)", "@range(u16,16)")
)
pe = (
    (base / "pe.csl")
    .read_text()
    .replace(".participants=8,", ".participants=16,")
    .replace("%8", "%16")
)
(r / "layout.csl").write_text(layout)
(r / "pe.csl").write_text(pe)
cmd = read(base / "sdk-command.json")
cmd = [v.replace("--fabric-dims=15,10", "--fabric-dims=23,18") for v in cmd]
schema = read(base / "schema.json")
schema.update(rows=16, cols=16)
inputs = []
expected = []
yy, xx = np.indices((16, 16))
words = lambda a: np.asarray(a, np.float16).view(np.uint16).tolist()
for e in range(8):
    a = np.full((16, 16, 4), -17.0)
    b = np.full((16, 16, 12), -19.0)
    c = np.full((16, 16, 6), -23.0)
    for k in range(3):
        a[:, :, k] = ((yy + 3 * xx + k + e) % 17 - 8) / 8
    for obj, length, axis in ((b, 11, 1), (c, 5, 0)):
        for k in range(length):
            obj[:, :, k] = (
                -(1 + ((xx if axis == 1 else yy) - e - k - 8) % 16) / 8
                - (yy if axis == 1 else xx) / 64
            )
    inputs.append(dict(A=a.tolist(), B=b.tolist(), C=c.tolist()))
    ar = a.copy()
    ar[:, :, :3] = np.sum(a[:, :, :3], axis=0, keepdims=True)
    d = b.copy()
    d[:, :, :11] = np.max(b[:, :, :11], axis=1, keepdims=True)
    cr = c.copy()
    cr[:, :, :5] = np.max(c[:, :, :5], axis=0, keepdims=True)
    delays = np.empty((16, 16, 4), int)
    for stage in range(4):
        delays[:, :, stage] = (xx * 11 + yy * 7 + e * 13 + stage * 17) % 31
        delays[(xx == (e + stage) % 16) & (yy == (e * 3 + stage) % 16), stage] += 127
    expected.append(
        dict(
            A=words(ar),
            B=words(b),
            C=words(cr),
            D=words(d),
            delays=delays.tolist(),
            delay_value=words(delays[:, :, 3:4]),
        )
    )
for n, v in [
    ("sdk-command.json", cmd),
    ("schema.json", schema),
    ("inputs.json", inputs),
    ("expected.json", expected),
]:
    (r / n).write_text(json.dumps(v) + "\n")
shutil.copyfile(__file__, r / "prepare.py")
(r / "provenance.json").write_text(
    json.dumps(
        dict(
            scope="16x16 primitive only: SUM Y then MAX X/Y, all-negative rotating remote winners including coordinates8..15, canaries, skew,8warm calls. No application qualification.",
            base_provenance_sha256=sha(base / "provenance.json"),
            files={f.name: sha(f) for f in r.iterdir() if f.is_file()},
        ),
        indent=2,
    )
    + "\n"
)
print(r.relative_to(ROOT))
