"""Scoped latency observations and unchanged numerical trajectory."""

import argparse, hashlib, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("plain", type=Path)
p.add_argument("timed", type=Path)
p.add_argument("report", type=Path)
a = p.parse_args()
assert not a.report.exists()
x = json.loads((a.plain / "results.json").read_text())
y = json.loads((a.timed / "results.json").read_text())
assert x["success"] and y["success"] and len(x["cases"]) == len(y["cases"]) == 8
rows = []
clock = lambda w: sum(v << (16 * i) for i, v in enumerate(w))
for e, (before, after) in enumerate(zip(x["cases"], y["cases"])):
    for name in before:
        assert before[name] == after[name], (e, name)
    phases = []
    for phase in range(3):
        d = []
        for row in after["timing"]:
            for words in row:
                v = words[phase * 6 : phase * 6 + 6]
                dt = (clock(v[3:]) - clock(v[:3])) % (1 << 48)
                assert 0 < dt < 1000000
                d.append(dt)
        phases.append(dict(min=min(d), max=max(d), values=d))
    rows.append(
        dict(
            epoch=e,
            ordinary_sum=phases[0],
            prescaled_mean=phases[1],
            opposite_axis_sum=phases[2],
        )
    )
files = [
    a.plain / "results.json",
    a.timed / "results.json",
    a.timed / "provenance.json",
    a.timed / "execution.json",
    a.timed / "pe.csl",
    Path(__file__),
]
r = dict(
    passed=True,
    epochs=8,
    all_original_raw_ports_unchanged=True,
    cycles=rows,
    scope="Per-PE start-to-callback simulator cycles including provider work and arrival waiting. Prior phases/axis/extent differ; this is not a speedup comparison or an end-to-end kernel benchmark. Timestamp calls perturb execution. Host I/O excluded.",
    files={str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
)
a.report.write_text(json.dumps(r, indent=2) + "\n")
print("PASS", [(v["epoch"], v["prescaled_mean"]["max"]) for v in rows])
