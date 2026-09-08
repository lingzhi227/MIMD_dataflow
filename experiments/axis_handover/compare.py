"""Compare two completed semantic probes without conflating their arithmetic."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import json
import sys
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "experiments"))
from probe_runtime import read, sha, verify


def compare(ring, planes, output):
    ring, planes, output = map(lambda p: Path(p).resolve(), (ring, planes, output))
    assert not output.exists()
    for root, review in [(ring, "handover-review.json"), (planes, "plane-review.json")]:
        verify(root)
        execution = read(root / "execution.json")
        report = read(root / review)
        assert execution["success"] and report["passed"]
        assert (
            execution["results_sha256"]
            == report["results_sha256"]
            == sha(root / "results.json")
        )
        assert report["provenance_sha256"] == sha(root / "provenance.json")
    assert (
        read(ring / "execution.json")["sdk_sha256"]
        == read(planes / "execution.json")["sdk_sha256"]
    )
    assert read(ring / "inputs.json") == read(planes / "inputs.json")
    assert read(ring / "runtime-options.json") == read(planes / "runtime-options.json")
    a, b = read(ring / "results.json"), read(planes / "results.json")
    assert a["success"] and b["success"] and len(a["cases"]) == len(b["cases"]) == 8
    ca = read(ring / "handover-review.json")["cycles_per_pe"]
    cb = read(planes / "plane-review.json")["cycles_per_pe"]
    rows = []
    for epoch, (x, y) in enumerate(zip(a["cases"], b["cases"])):
        for name in ["progress", "delays", "delay_value"]:
            np.testing.assert_array_equal(x[name], y[name])
        differing = sum(
            int(np.count_nonzero(np.asarray(x[k]) != np.asarray(y[k])))
            for k in ["A", "B", "C"]
        )
        ma, mb = int(np.max(ca[epoch])), int(np.max(cb[epoch]))
        rows.append(
            dict(
                epoch=epoch,
                ring_max_pe_cycles=ma,
                sdk_planes_max_pe_cycles=mb,
                sdk_to_ring_ratio=mb / ma,
                different_half_words=differing,
                ring_host_seconds=a["host_call_seconds"][epoch],
                sdk_host_seconds=b["host_call_seconds"][epoch],
            )
        )
    paths = [Path(__file__)]
    for root, review in [(ring, "handover-review.json"), (planes, "plane-review.json")]:
        paths += [
            root / n
            for n in [
                "provenance.json",
                "execution.json",
                "results.json",
                "inputs.json",
                "runtime-options.json",
                review,
            ]
        ]
    report = dict(
        passed=True,
        cases=rows,
        scope="Same64PE input, four explicit per-PE delays, four simulator threads and exported observation counts. Different grouped-half+control-ring versus native f32 SDK planes with final half narrowing. Whole-probe cycles include data, conversions, delay, waiting and protocol; not an isolated barrier saving, HLS application speedup, or hardware throughput.",
        total_different_half_words=sum(r["different_half_words"] for r in rows),
        total_compared_half_words=8 * 8 * 8 * 12,
        hashes={str(p.relative_to(ROOT)): sha(p) for p in paths},
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    compare(*sys.argv[1:])
