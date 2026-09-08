"""Compare generated RMS with explicitly corrected pinned source; immutable inputs."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, sys
from pathlib import Path
import numpy as np
from probe_runtime import verify, read, sha

p = argparse.ArgumentParser()
p.add_argument("hls", type=Path)
p.add_argument("control", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
sys.path.insert(0, str(a.hls / "implementation"))
from validate import audit

h = audit(a.hls)
verify(a.control)
ce = read(a.control / "execution.json")
c = read(a.control / "results.json")
assert (
    ce["success"]
    and ce["results_sha256"] == sha(a.control / "results.json")
    and c["success"]
    and c["runtime_instances"] == 1
)
hs = read(a.hls / "schedule.json")
cs = read(a.control / "schedule.json")
# The source control has no HLS samples; changing HLS sampling leaves the
# numerical schedule and ownership identical, but changes observation storage.
assert {
    k: v for k, v in hs.items() if k not in ("instrumentation", "memory_per_pe")
} == {k: v for k, v in cs.items() if k not in ("instrumentation", "memory_per_pe")}
assert read(a.hls / "runtime-options.json") == read(a.control / "runtime-options.json")
assert read(a.hls / "batches.json")[: len(c["cases"])] == read(
    a.control / "inputs.json"
)
r = read(a.hls / "results.json")
reports = []
for epoch, case in enumerate(c["cases"]):
    raw = np.asarray(case["hls_result"])
    np.testing.assert_array_equal(raw, r["diagnostics"][epoch]["result"])
    np.testing.assert_array_equal(case["hls_progress"], epoch + 1)
    t = np.asarray(case["hls_time"])
    assert (
        t.shape[-1] == 6
        and np.issubdtype(t.dtype, np.integer)
        and np.all((t >= 0) & (t < 65536))
    )
    cycles = sum((t[:, :, i + 3] - t[:, :, i]) * (1 << (16 * i)) for i in range(3))
    assert np.all((cycles > 0) & (cycles < 2**32))
    hc = h["cases"][epoch]["max_local_cycles"]
    cc = int(cycles.max())
    reports.append(
        dict(
            packed_half_bits_exact=True,
            hls_max_local_cycles=hc,
            corrected_source_max_local_cycles=cc,
            hls_to_source_ratio=hc / cc,
        )
    )
result = dict(
    passed=True,
    new_sdk_execution=False,
    scope="Generated RMS against pinned Prefill with explicit row-scale and feature-weight corrections. Same inputs, SDK runtime options and dimensions. HLS phase/progress wrapper differs. WSE3 simulator local intervals, not hardware throughput or unmodified upstream performance.",
    cases=reports,
    hashes={
        str(f): sha(f)
        for f in [
            a.hls / "results.json",
            a.hls / "manifest.json",
            a.control / "results.json",
            a.control / "provenance.json",
            Path(__file__),
        ]
    },
)
a.output.write_text(json.dumps(result, indent=2) + "\n")
print(a.output)
