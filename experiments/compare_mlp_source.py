"""Matched source/HLS rectangular MLP outputs, observers and scoped local cycles."""

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
g = read(a.source / "geometry.json")
assert all(s[k] == g[k] for k in ("M", "N", "F", "P"))
assert read(a.hls / "runtime-options.json") == read(a.source / "runtime-options.json")
prov = read(a.source / "provenance.json")
assert (
    prov["preserve_completed_left_ownership"]
    and s["instrumentation"] == prov["instrumentation"]
)
for option in ("--fabric-dims=", "--fabric-offsets="):
    hopt = [v for v in read(a.hls / "sdk-command.json") if v.startswith(option)]
    sopt = [v for v in read(a.source / "sdk-command.json") if v.startswith(option)]
    assert len(hopt) == 1 and hopt == sopt, "matched physical fabric required"
bs = read(a.hls / "batches.json")
sb = read(a.source / "logical-inputs.json")
blocked = s.get("down_accumulation") == "block_f32"
assert blocked == prov.get("blocked_down_accumulation", False)
source_calls = 8 if blocked else 3
assert bs[:source_calls] == sb
r = read(a.hls / "results.json")
c = read(a.source / "results.json")
assert (
    c["success"]
    and c["runtime_instances"] == 1
    and len(c["cases"]) == source_calls
    and len(r["cases"]) == (8 if blocked else 6)
)
rows = []
for epoch, (src, d) in enumerate(zip(c["cases"], r["diagnostics"])):
    np.testing.assert_array_equal(src["output"], d["result"])
    if blocked:
        np.testing.assert_array_equal(src["wide_accumulator"], d["wide_accumulator"])
    if s["instrumentation"] == "sampled":
        for src_key, hls_key in [
            ("gate", "gate_snapshot"),
            ("hidden", "hidden_snapshot"),
            ("activated_gate", "activated_gate"),
            ("up_history", "up_history"),
            ("gate_history", "gate_history"),
            ("down_history", "down_history"),
        ]:
            np.testing.assert_array_equal(src[src_key], d[hls_key])
        np.testing.assert_array_equal(
            src["gate_left_owner"],
            np.asarray(d["left_first"])[:, :, s["length"] : 2 * s["length"]],
        )
    np.testing.assert_array_equal(src["progress"], epoch + 1)
    t = np.asarray(src["timing"], np.int64)
    cycles = sum((t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)) % (
        1 << 48
    )
    assert np.all((cycles > 0) & (cycles < 2**32))
    cc = int(cycles.max())
    hc = h["cases"][epoch]["max_local_cycles"]
    rows.append(
        dict(
            target_half_bits_exact=True,
            internal_prefixes_exact=True if s["instrumentation"] == "sampled" else None,
            internal_prefixes_observed=s["instrumentation"] == "sampled",
            hls_max_local_cycles=hc,
            source_max_local_cycles=cc,
            hls_to_source_ratio=hc / cc,
        )
    )
a.output.write_text(
    json.dumps(
        dict(
            passed=True,
            blocked_down_accumulation=blocked,
            shared_precision_library=blocked,
            new_sdk_execution=False,
            cases=rows,
            scope="Supplied bounded X/U/G/D, identical logical inputs/geometry/SDK options. Source gate-left carry repair explicit. HLS immutable-input copies inside timed path, in-place hidden and dead-X output aliases; source uses destructive inputs/separate hidden. Sampled HLS adds first operands of all projections and activated-gate copy; source samples gate-left only. Counter omits half intermediate-prefix snapshots; blocked accumulation retains final f32 accumulator readout. Local WSE3 simulator intervals; no full-model or hardware claim.",
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
print(rows)
