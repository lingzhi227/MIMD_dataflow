"""Compare full CACHE ATTENTION runs against original Decode local vecmat arithmetic control."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
p.add_argument("source", type=Path)
p.add_argument("report", type=Path)
a = p.parse_args()
ROOT = repository_root(__file__)
bundle = a.bundle.resolve()
source = a.source.resolve()
assert not a.report.exists()
sys.path.insert(0, str(bundle / "implementation"))
from validate import audit
from mesh_cache_attention_sdk import packed

read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
review = audit(bundle)
assert review["passed"] and read(bundle / "qualification.json")["success"]
qualified_audit = read(bundle / "qualification.json")["audit"]
assert qualified_audit["passed"] and qualified_audit["epochs"] == review["epochs"]
# Host NumPy/BLAS may differ in the final bits of diagnostic f64 norms. Both
# independently satisfy the fixed gate; integer device-cycle evidence is exact.
assert [(c["epoch"], c["max_pe_cycles"]) for c in qualified_audit["cases"]] == [
    (c["epoch"], c["max_pe_cycles"]) for c in review["cases"]
]
provenance = read(source / "provenance.json")
for n, h in provenance["files"].items():
    assert sha(source / n) == h, n
upstream = ROOT / "third_party/sources/waferllm/Decode/src/decode.csl"
assert sha(upstream) == provenance["source_sha256"]
source_text = upstream.read_text()
functions = read(source / "source-functions.json")
assert set(functions) == {"gemv_static_step", "vecmat_computation"}
for name, body in functions.items():
    assert body.startswith("fn " + name + "(") and source_text.count(body) == 1
    assert (source / "source_vecmat.csl").read_text().count(body) == 1
assert provenance["hls_manifest_sha256"] == sha(bundle / "manifest.json")
assert read(source / "execution.json")["success"]
assert read(source / "execution.json")["results_sha256"] == sha(source / "results.json")
assert (
    read(source / "execution.json")["sdk_sha256"]
    == read(bundle / "qualification.json")["sdk_sha256"]
    == "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"
)
s, m, batches = [
    read(bundle / n) for n in ("schedule.json", "semantic.json", "batches.json")
]
a_result = read(bundle / "results.json")
b_result = read(source / "results.json")
source_inputs = read(source / "inputs.json")
assert (
    a_result["success"]
    and b_result["success"]
    and len(a_result["cases"])
    == len(b_result["cases"])
    == len(source_inputs)
    == len(batches)
    == 8
)
assert a_result["runtime_instances"] == b_result["runtime_instances"] == 1
assert read(source / "runtime-options.json") == read(bundle / "runtime-options.json")
rows = []


def cycles(d):
    t = np.asarray(d["timing"], np.uint64)
    v = (
        (t[:, :, 3] + (t[:, :, 4] << 16) + (t[:, :, 5] << 32))
        - (t[:, :, 0] + (t[:, :, 1] << 16) + (t[:, :, 2] << 32))
    ) & ((1 << 48) - 1)
    assert np.all((v > 0) & (v < 2**32))
    return int(v.max())


for epoch, (batch, expected_input, left, right) in enumerate(
    zip(batches, source_inputs, a_result["diagnostics"], b_result["cases"])
):
    actual_input = packed(s, m, batch)
    assert set(actual_input) == set(expected_input)
    for n, v in actual_input.items():
        np.testing.assert_array_equal(
            v, expected_input[n], err_msg=f"{epoch} source input {n}"
        )
    assert set(left) == set(right)
    ports = []
    for n, v in left.items():
        if n in ("timing", "queues"):
            continue
        np.testing.assert_array_equal(v, right[n], err_msg=f"{epoch} source {n}")
        ports.append(n)
    assert np.all((np.asarray(right["queues"]) & 60) == 60)
    l, r = cycles(left), cycles(right)
    rows.append(
        dict(
            epoch=epoch,
            exact_ports=ports,
            hls_max_pe_cycles=l,
            source_max_pe_cycles=r,
            ratio=l / r,
            hls_full_diagnostic_host_seconds=a_result["host_call_seconds"][epoch],
            source_full_diagnostic_host_seconds=b_result["host_call_seconds"][epoch],
        )
    )
report = dict(
    passed=True,
    epochs=8,
    checks=rows,
    hls_audit=review,
    scope="Original Decode gemv_static_step/vecmat_computation unchanged under same SDK X/Y f32 collective, repaired softmax, explicit local block merge and diagnostic schedule. Measures local projection lowering overhead only; not unmodified Decode performance or hardware throughput.",
    projection_flops=4 * s["B"] * s["N"] * s["S"] + 2 * s["B"] * s["N"] * s["N"],
    flop_scope="Three matrix projections only; denominator is entire CACHE ATTENTION max-PE interval, excludes host I/O; not roofline efficiency",
    files={
        str(p.resolve().relative_to(ROOT)): sha(p)
        for p in (
            bundle / "manifest.json",
            bundle / "results.json",
            bundle / "qualification.json",
            source / "provenance.json",
            source / "execution.json",
            source / "results.json",
            Path(__file__),
        )
    },
)
a.report.write_text(json.dumps(report, indent=2) + "\n")
print("FULL CACHE ATTENTION SOURCE COMPUTE COMPARISON PASS")
