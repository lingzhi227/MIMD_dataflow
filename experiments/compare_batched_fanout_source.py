"""Full standard HLS/source comparison including original-domain branch mathematics."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import json, subprocess, sys
from pathlib import Path
import numpy as np
from probe_runtime import verify, read, sha

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from batched_fanout_fixtures import check


def compare(bundle, source, output):
    bundle, source, output = map(lambda p: Path(p).resolve(), (bundle, source, output))
    assert not output.exists()
    verify(source)
    code = 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
    audit = json.loads(
        subprocess.check_output([sys.executable, "-c", code, str(bundle)], text=True)
    )
    assert audit["passed"]
    s = read(bundle / "schedule.json")
    bs = read(bundle / "batches.json")
    h = read(bundle / "results.json")
    r = read(source / "results.json")
    execution = read(source / "execution.json")
    q = read(bundle / "qualification.json")
    assert (
        execution["success"]
        and q["success"]
        and execution["results_sha256"] == sha(source / "results.json")
    )
    assert (
        execution["sdk_sha256"]
        == q["sdk_sha256"]
        == "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"
    )
    assert (
        h["success"]
        and r["success"]
        and h["runtime_instances"] == r["runtime_instances"] == 1
    )
    assert (
        bs == read(source / "logical-inputs.json")
        and len(bs) == len(h["diagnostics"]) == len(r["cases"]) == 8
    )
    # Decode source-control words with this run's verified transport contract.
    sys.path.insert(0, str(bundle / "implementation"))
    import mesh_batched_fanout_sdk as frozen_transport

    assert (
        Path(frozen_transport.__file__).resolve()
        == (bundle / "implementation/mesh_batched_fanout_sdk.py").resolve()
    )
    projection_flops = 2 * s["projections"] * s["B"] * s["N"] * s["F"]
    assert len(h["host_call_seconds"]) == len(r["host_call_seconds"]) == 8
    checks = []

    def ticks(row):
        t = np.asarray(row["timing"], np.int64)
        v = sum((t[:, :, i + 3] - t[:, :, i]) * (1 << (16 * i)) for i in range(3)) % (
            1 << 48
        )
        assert np.all((v > 0) & (v < 2**32))
        return v

    for e, (b, hs, ss) in enumerate(zip(bs, h["diagnostics"], r["cases"])):
        for name in ("X", "W", "result", "sums", "history", "partial", "projections"):
            np.testing.assert_array_equal(
                hs[name], ss[name], err_msg=f"{e} {name} source"
            )
        for k in range(s["projections"]):
            extent = s["Nt"] * s["Ft"]
            np.testing.assert_array_equal(
                np.asarray(hs["weights"])[:, :, k * extent : (k + 1) * extent],
                ss["weight" + str(k)],
                err_msg=f"{e} weight{k}",
            )
        m = read(bundle / "semantic.json")
        math = check(
            s["B"],
            s["N"],
            s["F"],
            s["projections"],
            b,
            frozen_transport.decode(s, m, ss),
        )
        hc, sc = ticks(hs), ticks(ss)
        checks.append(
            dict(
                epoch=e,
                exact_port_groups=7 + s["projections"],
                source_application_check=math,
                hls_max_pe_cycles=int(hc.max()),
                source_max_pe_cycles=int(sc.max()),
                ratio=float(hc.max() / sc.max()),
                projection_fma_flops=projection_flops,
                hls_projection_flops_per_max_pe_cycle=float(
                    projection_flops / hc.max()
                ),
                source_projection_flops_per_max_pe_cycle=float(
                    projection_flops / sc.max()
                ),
                hls_host_call_seconds=h["host_call_seconds"][e],
                source_host_call_seconds=r["host_call_seconds"][e],
                hls_cycles_per_pe=hc.tolist(),
                source_cycles_per_pe=sc.tolist(),
            )
        )
    dependencies = [
        bundle / "manifest.json",
        bundle / "results.json",
        bundle / "qualification.json",
        source / "provenance.json",
        source / "results.json",
        source / "execution.json",
        ROOT / "tests/support/batched_fanout_fixtures.py",
        Path(__file__),
    ]
    report = dict(
        passed=True,
        cases=checks,
        source_mathematics_eight_passed=True,
        operation_count_scope="Projection multiply/add operations only (two per FMA), excluding RMS; aggregate over the rectangle divided by max PE interval, not peak utilization or hardware throughput.",
        host_timing_scope="Original full input/launch/diagnostic output wall time including weights and witnesses; simulator host I/O, not production device latency.",
        min_ratio=min(c["ratio"] for c in checks),
        max_ratio=max(c["ratio"] for c in checks),
        hashes={str(p.relative_to(ROOT)): sha(p) for p in dependencies},
        scope="Precision/order-matched repaired pinned Decode normalized projections and original fused grouped collective, all eight original fixtures and replicas. Simulator max-PE cycles including observer work, excluding host I/O; no hardware/full Decode throughput.",
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output, report["min_ratio"], report["max_ratio"])


if __name__ == "__main__":
    compare(*sys.argv[1:])
