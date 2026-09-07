"""Matched resident HLS/source composition, with actual frozen implementation audit."""

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
execution = read(a.source / "execution.json")
assert execution["success"] and execution["results_sha256"] == sha(
    a.source / "results.json"
)
s = read(a.hls / "schedule.json")
rs = read(a.source / "schedule.json")
assert (s["M"], s["N"], s["P"], s["epsilon"]) == (
    rs["M"],
    rs["N"],
    rs["cols"],
    rs["epsilon"],
) and rs["rows"] == rs["cols"]
assert read(a.hls / "runtime-options.json") == read(a.source / "runtime-options.json")
assert read(a.hls / "batches.json")[: len(read(a.source / "inputs.json"))] == read(
    a.source / "inputs.json"
)
r = read(a.hls / "results.json")
c = read(a.source / "results.json")
assert (
    c["success"]
    and c["runtime_instances"] == 1
    and len(c["cases"]) == len(read(a.source / "inputs.json"))
    and len(c["cases"]) in (2, 6)
    and len(r["cases"]) == 6
)
cases = []
for i, (source, d) in enumerate(zip(c["cases"], r["diagnostics"])):
    np.testing.assert_array_equal(source["hls_result"], d["result"])
    if s["instrumentation"] == "sampled":
        np.testing.assert_array_equal(source["hls_normalized"], d["normalized"])
    np.testing.assert_array_equal(source["hls_progress"], i + 1)
    t = np.asarray(source["hls_time"], np.int64)
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
            scope="Resident corrected RMSNorm -> original Q projection versus HLS composition. Matching inputs, precision, geometry, runtime options and pinned source. Sampled HLS has vector normalized copy and prefixes; sampled source has a scalar normalized copy and no prefixes, so its interval ratio includes unequal observation costs. Counter source/HLS omit tensor observations. Local WSE3 simulator intervals, no hardware claim.",
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
