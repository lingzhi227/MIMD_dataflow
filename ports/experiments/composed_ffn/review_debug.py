"""Verify the public stage debugger against preserved actual SDK words."""

import argparse, hashlib, json, struct, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
p.add_argument("snapshot", type=Path)
p.add_argument("report", type=Path)
a = p.parse_args()
assert not a.report.exists()
sys.path.insert(0, str(a.bundle.resolve() / "implementation"))
from integrity import verify_bundle, verify_codegen
from projected_cache_ffn_debug import inspect, STAGES
from projected_cache_ffn_reference import audit_cases

verify_bundle(a.bundle)
verify_codegen(a.bundle)
s, m, bs = [
    json.loads((a.bundle / n).read_text())
    for n in ("schedule.json", "semantic.json", "batches.json")
]
r = json.loads(a.snapshot.read_text())
audit = audit_cases(s, m, bs, r, require_complete=False)
count = len(r["cases"])
rows = []
for e in range(count):
    for x, y in (
        (0, 0),
        (0, 1),
        (0, 15),
        (1, 0),
        (1, 1),
        (1, 15),
        (15, 0),
        (15, 1),
        (15, 15),
    ):
        for step, port in enumerate(STAGES):
            v = inspect(s, r, f"p{x}_{y}", e, step)
            raw = r["diagnostics"][e][port][y][x]
            assert v["available"] and v["observed"] and v["raw_words"] == raw
            assert v["values"] == [
                struct.unpack("<e", struct.pack("<H", w))[0] for w in raw
            ]
            rows.append(dict(epoch=e, node=v["node"], stage=port, words=len(raw)))
if count < m["epochs"]:
    missing = inspect(s, r, "p15_15", count, 0)
    assert (
        not missing["available"]
        and missing["values"] is None
        and missing["raw_words"] is None
    )
for step in range(len(STAGES)):
    v = inspect(s, None, "p0_0", 0, step)
    assert not v["available"] and not v["observed"]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
a.report.write_text(
    json.dumps(
        dict(
            passed=True,
            scope="Public frozen stage inspector versus archived actual words on nine PE classes and every saved call; missing calls are never synthesized. Partial snapshots do not qualify a run.",
            audit=audit,
            views=rows,
            files={
                str(p.resolve()): sha(p)
                for p in (
                    a.snapshot,
                    a.bundle / "manifest.json",
                    a.bundle / "implementation/projected_cache_ffn_debug.py",
                    Path(__file__),
                )
            },
        ),
        indent=2,
    )
    + "\n"
)
print("DEBUG SNAPSHOT PASS", len(rows))
