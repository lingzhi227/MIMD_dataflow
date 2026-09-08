"""Measured source precision-plus-observation cost; unlike compiler overhead."""

import argparse, json
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify

p = argparse.ArgumentParser()
p.add_argument("down_only", type=Path)
p.add_argument("all_block", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
for root in (a.down_only, a.all_block):
    verify(root)
    e = read(root / "execution.json")
    assert e["success"] and e["results_sha256"] == sha(root / "results.json")
assert not read(a.down_only / "provenance.json").get(
    "blocked_upper_accumulation", False
)
assert read(a.all_block / "provenance.json")["blocked_upper_accumulation"]
for name in (
    "logical-inputs.json",
    "geometry.json",
    "sdk-command.json",
    "runtime-options.json",
):
    assert read(a.down_only / name) == read(a.all_block / name)
rs = [read(root / "results.json") for root in (a.down_only, a.all_block)]
assert all(r["success"] and len(r["cases"]) == 3 for r in rs)
rows = []
for epoch, (d, b) in enumerate(zip(rs[0]["cases"], rs[1]["cases"])):

    def cycles(v):
        t = np.asarray(v["timing"], np.int64)
        c = sum((t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)) % (
            1 << 48
        )
        assert np.all((c > 0) & (c < 2**32))
        return int(c.max())

    dc, bc = cycles(d), cycles(b)
    rows.append(
        dict(
            epoch=epoch,
            down_only_max_local_cycles=dc,
            all_block_max_local_cycles=bc,
            all_block_to_down_only_ratio=bc / dc,
        )
    )
a.output.write_text(
    json.dumps(
        dict(
            passed=True,
            new_sdk_execution=False,
            cases=rows,
            scope="Two executed adapted source controls, identical first three inputs/fabric/options. Additional up/gate block-f32 merges AND retained f32 snapshots; their costs are not separated. Numerical policies differ, so bitwise equivalence is not asserted. This is not HLS compiler overhead, not eight-call qualification of the down-only policy, and not hardware throughput.",
            hashes={
                str(f): sha(f)
                for root in (a.down_only, a.all_block)
                for f in (root / "provenance.json", root / "results.json")
            },
        ),
        indent=2,
    )
    + "\n"
)
print(a.output, rows)
