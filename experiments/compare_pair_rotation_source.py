"""Compare frozen HLS and isolated pinned source gated activation on matched inputs."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify

p = argparse.ArgumentParser()
p.add_argument("hls", type=Path)
p.add_argument("source", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
code = 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
h = json.loads(
    subprocess.check_output(
        [sys.executable, "-c", code, str(a.hls.resolve())], text=True
    )
)
assert h["passed"]
verify(a.source)
e = read(a.source / "execution.json")
assert e["success"] and e["results_sha256"] == sha(a.source / "results.json")
s = read(a.hls / "schedule.json")
assert s == read(a.source / "schedule.json")
assert read(a.hls / "runtime-options.json") == read(a.source / "runtime-options.json")
b = read(a.source / "batches.json")
assert read(a.hls / "batches.json")[: len(b)] == b
r = read(a.hls / "results.json")
c = read(a.source / "results.json")
assert (
    c["success"]
    and c["runtime_instances"] == 1
    and c["launches"] == ["hls_main"] * len(b)
)
assert (
    len(c["cases"]) == len(c["diagnostics"]) == len(b)
    and len(b) in (2, 3, 6)
    and len(r["cases"]) == 6
)
cases = []
for i, (d, hd) in enumerate(zip(c["diagnostics"], r["diagnostics"])):
    sys.path.insert(0, str(a.hls.resolve() / "implementation"))
    from mesh_pair_rotation_sdk import extents
    from mesh_common import unpack_tiles

    for name, n in extents(s).items():
        v = np.asarray(d[name])
        assert (
            v.shape == (s["rows"], s["cols"], n)
            and np.issubdtype(v.dtype, np.integer)
            and np.all((v >= 0) & (v < 65536))
        )
    for name in ("cosine", "sine", "result", "history"):
        np.testing.assert_array_equal(d[name], hd[name])
    np.testing.assert_array_equal(d["x"], hd["result"])
    np.testing.assert_array_equal(
        d["progress"], np.tile([s["Nt"] // 2, 1, i + 1], (s["rows"], s["cols"], 1))
    )
    output = (
        unpack_tiles(
            np.asarray(d["result"], np.uint16).view(np.float16), s["Mt"], s["Nt"], "F"
        )
        .astype(float)
        .ravel()
        .tolist()
    )
    assert c["cases"][i] == {"rotated": output}
    t = np.asarray(d["timing"], np.int64)
    cycles = sum((t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)) % (
        1 << 48
    )
    assert np.all((cycles > 0) & (cycles < 2**32))
    cc = int(cycles.max())
    hc = h["cases"][i]["max_local_cycles"]
    cases.append(
        dict(
            target_half_bits_exact=True,
            hls_max_local_cycles=hc,
            source_max_local_cycles=cc,
            hls_to_source_ratio=hc / cc,
        )
    )
a.output.write_text(
    json.dumps(
        dict(
            passed=True,
            new_sdk_execution=False,
            scope="Original xq_rope arithmetic/DSD loop with row-sized scratch and empty completion callback; shared frozen transport. Source overwrites x; HLS preserves x and writes separate output. Product observations matched. HLS tracks per-pair progress; source assigns completed count. Local WSE3 simulator intervals; no full RoPE model/inference/hardware claim.",
            cases=cases,
            hashes={
                str(v): sha(v)
                for v in [
                    a.hls / "manifest.json",
                    a.hls / "results.json",
                    a.source / "provenance.json",
                    a.source / "results.json",
                    Path(__file__),
                ]
            },
        ),
        indent=2,
    )
    + "\n"
)
print(a.output)
print(cases)
