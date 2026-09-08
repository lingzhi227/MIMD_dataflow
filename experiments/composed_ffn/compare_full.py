"""Compare complete actual HLS/source-compute runs; refuse partial evidence."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path)
    p.add_argument("source", type=Path)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    assert not a.report.exists()
    bundle, source = a.bundle.resolve(), a.source.resolve()
    sys.path.insert(0, str(bundle / "implementation"))
    from mesh_projected_cache_ffn_sdk import audit, packed

    read = lambda path: json.loads(path.read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    baseline = audit(bundle)
    assert baseline["complete"] and baseline["completed_calls"] == 8
    provenance = read(source / "provenance.json")
    for name, digest in provenance["files"].items():
        assert sha(source / name) == digest, name
    assert provenance["hls_manifest_sha256"] == sha(bundle / "manifest.json")
    upstream = ROOT / "third_party/sources/waferllm/Decode/src/decode.csl"
    assert sha(upstream) == provenance["upstream_sha256"]
    text = upstream.read_text()
    for name, digest in provenance["original_functions_sha256"].items():
        start = re.search(r"^fn " + name + r"\(", text, re.M).start()
        end = text.index("{", start) + 1
        depth = 1
        while depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        body = text[start:end]
        assert hashlib.sha256(body.encode()).hexdigest() == digest
        assert (source / "source_vecmat.csl").read_text().count(body) == 1
    for path in bundle.glob("*.csl"):
        expected = path.read_text()
        if path.name == "batched_matmul_blocked.csl":
            assert expected.count('"batched_matmul_local.csl"') == 2
            expected = expected.replace(
                '"batched_matmul_local.csl"', '"source_vecmat.csl"'
            )
        assert (source / path.name).read_text() == expected, path.name
    s, m, batches, left, right = [
        read(path)
        for path in (
            bundle / "schedule.json",
            bundle / "semantic.json",
            bundle / "batches.json",
            bundle / "results.json",
            source / "results.json",
        )
    ]
    assert left["success"] is right["success"] is True
    assert (
        len(left["diagnostics"])
        == len(left["cases"])
        == len(right["cases"])
        == len(batches)
        == 8
    )
    assert type(left["runtime_instances"]) is type(right["runtime_instances"]) is int
    assert left["runtime_instances"] == right["runtime_instances"] == 1
    assert left["launches"] == ["hls_main"] * 8
    assert read(source / "runtime-options.json") == read(
        bundle / "runtime-options.json"
    )
    for base in (bundle, source):
        execution = read(base / "execution.json")
        assert execution["success"] is True and execution["results_sha256"] == sha(
            base / "results.json"
        )
        assert (
            execution["sdk_sha256"]
            == "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"
        )
    packed_inputs = read(source / "inputs.json")
    assert len(packed_inputs) == 8
    rows = []

    def cycles(raw):
        original = np.asarray(raw["timing"])
        assert original.shape == (16, 16, 6) and np.issubdtype(
            original.dtype, np.integer
        )
        assert np.all((original >= 0) & (original < 65536))
        t = original.astype(np.uint64)
        ticks = lambda x: x[:, :, 0] + (x[:, :, 1] << 16) + (x[:, :, 2] << 32)
        value = (ticks(t[:, :, 3:]) - ticks(t[:, :, :3])) & np.uint64((1 << 48) - 1)
        assert np.all((value > 0) & (value < 1 << 40))
        return int(value.max())

    for epoch, (batch, want, x, y) in enumerate(
        zip(batches, packed_inputs, left["diagnostics"], right["cases"])
    ):
        actual = packed(s, m, batch)
        assert set(actual) == set(want)
        for name in actual:
            np.testing.assert_array_equal(
                actual[name], want[name], err_msg=f"{epoch} input {name}"
            )
        assert set(x) == set(y)
        for name in y:
            raw = np.asarray(y[name])
            assert raw.shape == np.asarray(x[name]).shape
            assert np.issubdtype(raw.dtype, np.integer)
            assert np.all((raw >= 0) & (raw < 65536))
        for name in set(x) - {"timing"}:
            np.testing.assert_array_equal(
                x[name], y[name], err_msg=f"{epoch} raw {name}"
            )
        hls, control = cycles(x), cycles(y)
        rows.append(
            dict(
                epoch=epoch,
                identical_non_timing_ports=len(x) - 1,
                hls_cycles=hls,
                source_compute_cycles=control,
                hls_over_source=hls / control,
            )
        )
    files = [
        bundle / name
        for name in (
            "manifest.json",
            "results.json",
            "execution.json",
            "runtime-options.json",
        )
    ]
    files += [
        source / name
        for name in (
            "provenance.json",
            "results.json",
            "execution.json",
            "runtime-options.json",
            "inputs.json",
            "source_vecmat.csl",
        )
    ]
    files += [upstream, Path(__file__)]
    report = dict(
        passed=True,
        epochs=8,
        checks=rows,
        frozen_audit=baseline,
        files={str(path.relative_to(ROOT)): sha(path) for path in files},
        scope="Same full 35-node resident HLS graph and inputs; nine local contractions replaced by pinned Decode bodies with initialization timed. Other compute modules and SDK communication shared. Instrumented simulator PE-cycle comparison only, not original full Decode, hardware performance or host I/O throughput.",
    )
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print("FULL SOURCE COMPARISON PASS", flush=True)


if __name__ == "__main__":
    main()
