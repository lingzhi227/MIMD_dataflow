"""Check resident source RMS/projection against staged and standard references."""

import argparse, json, sys, math
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from mesh_rms import reference
from mesh_twohop import block_index
from mesh_common import unpack_tiles

p = argparse.ArgumentParser()
p.add_argument("probe", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
verify(a.probe)
execution = read(a.probe / "execution.json")
assert execution["success"] and execution["results_sha256"] == sha(
    a.probe / "results.json"
)
s = read(a.probe / "schedule.json")
r = read(a.probe / "results.json")
batches = read(a.probe / "inputs.json")
assert len(batches) == len(r["cases"]) == 6 and r["runtime_instances"] == 1
m, n, pe, mt, nt = s["M"], s["N"], s["cols"], s["Mt"], s["Nt"]
cases = []
for epoch, (b, c) in enumerate(zip(batches, r["cases"])):
    x = np.asarray(b["x"], float).reshape(m, n)
    w = np.asarray(b["w"], float)
    q = np.asarray(b["q"], float).reshape(n, n)
    _, _, _, normalized = reference(s, x, w)
    observed = {}
    for name in ("hls_normalized", "hls_result"):
        raw = np.asarray(c[name])
        assert raw.shape == (pe, pe, mt * nt) and np.issubdtype(raw.dtype, np.integer)
        assert np.all((raw >= 0) & (raw < 65536))
        observed[name] = unpack_tiles(
            raw.astype(np.uint16).view(np.float16), mt, nt, "F"
        ).astype(float)
    expected = np.zeros((m, n))
    for y in range(pe):
        for col in range(pe):
            tile = np.zeros((mt, nt))
            for step in range(pe):
                block = block_index(pe, y, col, step)
                for k in range(block * nt, (block + 1) * nt):
                    tile = np.asarray(
                        tile
                        + normalized[y * mt : (y + 1) * mt, k, None]
                        * q[None, k, col * nt : (col + 1) * nt],
                        np.float16,
                    ).astype(float)
            expected[y * mt : (y + 1) * mt, col * nt : (col + 1) * nt] = tile

    def bits(v):
        return np.asarray(v, np.float16).view(np.uint16)

    mismatches = {
        "normalization": int(
            np.count_nonzero(bits(normalized) != bits(observed["hls_normalized"]))
        ),
        "projection": int(
            np.count_nonzero(bits(expected) != bits(observed["hls_result"]))
        ),
    }
    standard_norm = x * w / np.sqrt(np.sum(x * x, axis=1)[:, None] / n + 1e-6)
    standard = standard_norm @ q
    error = observed["hls_result"] - standard
    relative = float(np.linalg.norm(error) / max(np.linalg.norm(standard), 1e-30))
    maximum = float(np.max(np.abs(error)))
    fixed = relative <= 0.015 and maximum <= 0.02 * max(
        float(np.max(np.abs(standard))), 1e-30
    )
    np.testing.assert_array_equal(c["hls_progress"], epoch + 1)
    t = np.asarray(c["hls_time"], np.int64)
    cycles = sum((t[:, :, i + 3] - t[:, :, i]) * (1 << (16 * i)) for i in range(3)) % (
        1 << 48
    )
    assert np.all((cycles > 0) & (cycles < 2**32))
    cases.append(
        dict(
            passed=not any(mismatches.values()) and fixed,
            bit_mismatches=mismatches,
            standard_relative_l2=relative,
            standard_max_abs_error=maximum,
            standard_accuracy_passed=fixed,
            max_local_cycles=int(cycles.max()),
        )
    )
report = dict(
    passed=all(c["passed"] for c in cases),
    scope="Corrected original Prefill resident RMSNorm then Q projection. No intermediate host computation or transfer; staged target bit oracle and standard mathematical composition. Not HLS qualification or hardware performance.",
    cases=cases,
    hashes={
        str(v): sha(v)
        for v in [
            a.probe / "provenance.json",
            a.probe / "results.json",
            Path(__file__),
            ROOT / "toolchain/mesh_rms.py",
            ROOT / "toolchain/mesh_twohop.py",
            ROOT / "toolchain/sdk_math_reference.py",
        ]
    },
)
a.output.write_text(json.dumps(report, indent=2) + "\n")
print(a.output, report["passed"])
print(cases)
assert report["passed"]
