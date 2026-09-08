"""Check differing outputs against source-permitted train accumulation orders.

Uses original CSC entries only. This is a bounded rounding explanation, not a
memory/protocol proof or a relaxation of the fixed original-entry accuracy test.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, math, hashlib
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("hls", type=Path)
p.add_argument("native", type=Path)
a = p.parse_args()
read = lambda p: json.loads(p.read_text())
s = read(a.hls / "schedule.json")
m = read(a.hls / "semantic.json")
batches = read(a.hls / "batches.json")
h = read(a.hls / "results.json")
n = read(a.native / "results.json")
H, W = s["rows"], s["cols"]
g = s["geometry"]
br, bc, lx, ly = [
    g[k] for k in ("block_rows", "block_cols", "local_vec_sz", "local_out_vec_sz")
]
keys = [x["host"] for x in m["nodes"][:4]]
outkey = m["nodes"][-1]["host"]


def f(x):
    return float(np.float32(x))


def fold(entries, x, fused):
    acc = 0.0
    for c, v in entries:
        product = float(v) * float(x[c])
        acc = f(acc + product) if fused else f(acc + f(product))
    return acc


report = []
for ep, (b, hc, nc) in enumerate(zip(batches, h["cases"], n["cases"])):
    hv = np.asarray(hc[outkey], np.float32)
    nv = np.asarray(nc[outkey], np.float32)
    changed = np.flatnonzero(hv.view(np.uint32) != nv.view(np.uint32))
    values, indices, offsets, x = [b[k] for k in keys]
    wanted = set(map(int, changed))
    entries = {r: [[] for _ in range(W)] for r in wanted}
    for c in range(s["N"]):
        for q in range(offsets[c], offsets[c + 1]):
            r = indices[q]
            if r in wanted:
                entries[r][c // bc].append((c, values[q]))
    cases = []
    for r in changed:
        r = int(r)
        prow = r // br
        owner = (r % br) // ly
        options = []
        for pc, part in enumerate(entries[r]):
            options.append(set())
            for split in (pc * bc + prow * lx, pc * bc + (prow + 1) * lx):
                north = sorted((e for e in part if e[0] >= split), reverse=True)
                south = sorted(e for e in part if e[0] < split)
                for fused in (False, True):
                    options[-1].add(f(fold(north, x, fused) + fold(south, x, fused)))
        # Each train forwards nearest partition first, maintaining its own order.
        west = list(range(owner + 1, W))
        east = list(range(owner - 1, -1, -1))
        states = {(0, 0): options[owner]}
        for total in range(len(west) + len(east) + 1):
            for i in range(len(west) + 1):
                j = total - i
                if not 0 <= j <= len(east) or (i, j) not in states:
                    continue
                cur = states[i, j]
                for ni, nj, pc in (
                    (i + 1, j, west[i] if i < len(west) else None),
                    (i, j + 1, east[j] if j < len(east) else None),
                ):
                    if pc is not None:
                        states.setdefault((ni, nj), set()).update(
                            f(v + q) for v in cur for q in options[pc]
                        )
        allowed = states[len(west), len(east)]
        cases.append(
            dict(
                row=r,
                hls=float(hv[r]),
                native=float(nv[r]),
                allowed_count=len(allowed),
                hls_in_allowed=float(hv[r]) in allowed,
                native_in_allowed=float(nv[r]) in allowed,
            )
        )
    report.append(dict(epoch=ep, changed=len(changed), rows=cases))
passed = all(
    v["hls_in_allowed"] and v["native_in_allowed"] for e in report for v in e["rows"]
)
result = dict(
    passed=passed,
    scope="Differing rows only; original entries, north/south local assignment alternatives, fixed directional order and all interleavings of serialized east/west accumulation. Includes separate and fused local multiply-add permitted by relaxed policy; not an instruction-level proof of which sequence ran.",
    hls_results_sha256=hashlib.sha256(
        (a.hls / "results.json").read_bytes()
    ).hexdigest(),
    native_results_sha256=hashlib.sha256(
        (a.native / "results.json").read_bytes()
    ).hexdigest(),
    epochs=report,
)
with (a.native / "rounding-orders.json").open("x") as stream:
    json.dump(result, stream, indent=2)
print(
    json.dumps(
        dict(
            passed=passed,
            changed=[e["changed"] for e in report],
            unexplained=[
                v
                for e in report
                for v in e["rows"]
                if not v["hls_in_allowed"] or not v["native_in_allowed"]
            ],
        )
    )
)
if not passed:
    raise SystemExit(1)
