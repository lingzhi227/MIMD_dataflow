"""Matched original score schedule versus generated HLS, with observer scope."""

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
g = read(a.source / "geometry.json")
assert (s["M"], s["N"], s["P"]) == (g["M"], g["N"], g["P"])
assert read(a.hls / "runtime-options.json") == read(a.source / "runtime-options.json")
assert s["instrumentation"] == read(a.source / "provenance.json").get(
    "instrumentation", "sampled"
)
sys.path.insert(0, str(a.hls.resolve() / "implementation"))
from mesh_score_sdk import packed
from mesh_score import inputs

r = read(a.hls / "results.json")
c = read(a.source / "results.json")
bs = read(a.hls / "batches.json")
sb = read(a.source / "inputs.json")
m = read(a.hls / "semantic.json")
assert (
    c["success"]
    and c["runtime_instances"] == 1
    and len(c["cases"]) == len(sb) == 3
    and len(r["cases"]) == 6
)
cases = []
for epoch, (src, d, b) in enumerate(zip(c["cases"], r["diagnostics"], sb)):
    for name, v in packed(s, inputs(m, bs[epoch])).items():
        np.testing.assert_array_equal(v, b[name])
    np.testing.assert_array_equal(src["score"], d["result"])
    np.testing.assert_array_equal(src["history"], d["history"])
    np.testing.assert_array_equal(src["q"], d["q"])
    np.testing.assert_array_equal(src["progress"], epoch + 1)
    t = np.asarray(src["timing"], np.int64)
    cycles = sum((t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)) % (
        1 << 48
    )
    assert np.all((cycles > 0) & (cycles < 2**32))
    cc = int(cycles.max())
    hc = h["cases"][epoch]["max_local_cycles"]
    cases.append(
        dict(
            target_half_bits_exact=True,
            partial_history_exact=True if s["instrumentation"] == "sampled" else None,
            partial_history_observed=s["instrumentation"] == "sampled",
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
            scope="Original QK-transpose vertical exchange and rotating-root reduction, same inputs/geometry/runtime options. HLS copies immutable K into private working storage and records per-round roots/progress. Sampled HLS additionally observes each live K owner; source only observes partials. Counter mode omits both tensor histories. Local WSE3 simulator intervals, no full attention/hardware claim.",
            cases=cases,
            hashes={
                str(x): sha(x)
                for x in [
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
