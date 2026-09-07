"""Compare frozen HLS and isolated pinned source gated activation on matched inputs."""

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
assert len(c["cases"]) == len(c["diagnostics"]) == len(b) == 2 and len(r["cases"]) == 6
cases = []
for i, (d, hd) in enumerate(zip(c["diagnostics"], r["diagnostics"])):
    for name, n in [
        ("up", s["length"]),
        ("gate", s["length"]),
        ("result", s["length"]),
        ("activated", s["length"] if s["instrumentation"] == "sampled" else 1),
        ("progress", 3),
        ("timing", 6),
    ]:
        v = np.asarray(d[name])
        assert (
            v.shape == (s["rows"], s["cols"], n)
            and np.issubdtype(v.dtype, np.integer)
            and np.all((v >= 0) & (v < 65536))
        )
    np.testing.assert_array_equal(d["up"], hd["up"])
    np.testing.assert_array_equal(d["result"], hd["result"])
    np.testing.assert_array_equal(
        d["progress"], np.tile([1, 1, i + 1], (s["rows"], s["cols"], 1))
    )
    if s["instrumentation"] == "sampled":
        np.testing.assert_array_equal(d["gate"], hd["activated"])
        np.testing.assert_array_equal(d["activated"], hd["activated"])
    # Source gate is intentionally transformed in place; reconstruct its exact value independently.
    sys.path.insert(0, str(a.hls.resolve() / "implementation"))
    from mesh_swiglu import reference, inputs
    from mesh_common import pack_tiles, unpack_tiles

    act, _ = reference(*inputs(read(a.hls / "semantic.json"), b[i]))
    np.testing.assert_array_equal(
        d["gate"],
        pack_tiles(act, s["rows"], s["cols"], "F").astype(np.float16).view(np.uint16),
    )
    output = (
        unpack_tiles(
            np.asarray(d["result"], np.uint16).view(np.float16), s["Mt"], s["Nt"], "F"
        )
        .astype(float)
        .ravel()
        .tolist()
    )
    assert c["cases"][i] == {"gated": output}
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
            scope="Isolated pinned source SiLU/product bodies with empty completion callback; shared frozen transport. Source overwrites gate; HLS preserves inputs and reuses private activation. Sampled HLS adds activation copy, source aliases it. Counter mode omits that copy. Local WSE3 simulator intervals; no full Prefill/hardware claim.",
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
