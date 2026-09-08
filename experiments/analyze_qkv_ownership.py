"""Test branch-boundary physical ownership from actual live/named-buffer snapshots."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from mesh_rms import reference
from mesh_twohop import block_index

p = argparse.ArgumentParser()
p.add_argument("original", type=Path)
p.add_argument("repaired", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
reports = []
for root, repaired in [(a.original, False), (a.repaired, True)]:
    verify(root)
    e = read(root / "execution.json")
    assert e["success"] and e["results_sha256"] == sha(root / "results.json")
    prov = read(root / "provenance.json")
    assert prov["preserve_completed_left_ownership"] == repaired
    s = read(root / "schedule.json")
    r = read(root / "results.json")
    bs = read(root / "inputs.json")
    assert len(r["cases"]) == len(bs) == 2
    pe, mt, nt = s["cols"], s["Mt"], s["Nt"]
    epochs = []
    for b, c in zip(bs, r["cases"]):
        norm = reference(
            s, np.asarray(b["x"]).reshape(s["M"], s["N"]), np.asarray(b["w"])
        )[-1]
        expected = np.zeros((pe, pe, mt * nt), np.uint16)
        for y in range(pe):
            for x in range(pe):
                k = block_index(pe, y, x, 0)
                expected[y, x] = (
                    norm[y * mt : (y + 1) * mt, k * nt : (k + 1) * nt]
                    .ravel(order="F")
                    .astype(np.float16)
                    .view(np.uint16)
                )
        fields = {}
        for name in ("live_trace", "norm_trace", "scratch_trace"):
            raw = np.asarray(c[name])
            assert (
                raw.shape == (pe, pe, 3 * mt * nt)
                and np.issubdtype(raw.dtype, np.integer)
                and np.all((raw >= 0) & (raw < 65536))
            )
            raw = raw.astype(np.uint16).reshape(pe, pe, 3, mt * nt)
            matches = np.all(raw == expected[:, :, None, :], axis=-1)
            fields[name] = dict(
                matches_by_stage=matches.transpose(2, 0, 1).tolist(),
                wrong_pes_per_stage=np.count_nonzero(~matches, axis=(0, 1)).tolist(),
                wrong_words_per_stage=np.count_nonzero(
                    raw != expected[:, :, None, :], axis=(0, 1, 3)
                ).tolist(),
            )
        assert fields["live_trace"]["wrong_pes_per_stage"][0] == 0
        if repaired:
            assert fields["live_trace"]["wrong_pes_per_stage"] == [0, 0, 0]
        else:
            assert fields["live_trace"]["wrong_pes_per_stage"][1] > 0
        epochs.append(fields)
    reports.append(
        dict(
            bundle=str(root),
            pointer_roles_preserved=repaired,
            epochs=epochs,
            results_sha256=sha(root / "results.json"),
        )
    )
a.output.write_text(
    json.dumps(
        dict(
            passed=True,
            scope="Actual completed live-send data versus named normalized/scratch storage at each projection boundary. Expected aligned block comes from independently replayed source RMS and two-hop initial block rule. This audit validates ownership hypotheses, not numerical QKV results (separate audit).",
            cases=reports,
            analyzer_sha256=sha(Path(__file__)),
        ),
        indent=2,
    )
    + "\n"
)
print(a.output)
print(
    [
        (
            r["pointer_roles_preserved"],
            [e["live_trace"]["wrong_pes_per_stage"] for e in r["epochs"]],
        )
        for r in reports
    ]
)
