"""Compare matched grouped GEMV diagnostic modes, preserving precision and topology."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np


def read(p):
    return json.loads(p.read_text())


def compare(full, lean):
    for root in (full, lean):
        assert read(root / "qualification.json")["success"]
        subprocess.run(
            [sys.executable, str(root / "implementation/validate.py"), str(root)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    for name in ["source.cpp", "batches.json", "runtime-options.json"]:
        assert (full / name).read_bytes() == (lean / name).read_bytes(), name
    a, b = read(full / "semantic.json"), read(lean / "semantic.json")
    a.pop("instrumentation", None)
    b.pop("instrumentation", None)
    assert a == b, "same mathematical and spatial program required"
    assert read(lean / "schedule.json")["instrumentation"] == "counters"
    assert read(full / "schedule.json").get("instrumentation", "sampled") == "sampled"
    fr, lr = read(full / "results.json"), read(lean / "results.json")
    assert (
        fr["cases"] == lr["cases"]
    ), "output bit-representable values must be identical"
    rows = []
    for epoch, (f, l) in enumerate(zip(fr["diagnostics"], lr["diagnostics"])):
        assert l["history_bits"] is None, "do not claim missing history"
        assert np.all(np.asarray(l["timing"])[:, :, 6:] == 0), "compute timing omitted"
        for name in ["progress", "queue", "result_bits"]:
            assert f[name] == l[name], name
        ticks = []
        for d in (f, l):
            t = np.asarray(d["timing"], np.int64)
            v = sum((t[:, :, i + 3] - t[:, :, i]) * (1 << (16 * i)) for i in range(3))
            assert np.all((v > 0) & (v < 2**32))
            ticks.append(int(v.max()))
        rows.append(
            dict(
                epoch=epoch,
                sampled_max_local_cycles=ticks[0],
                counters_max_local_cycles=ticks[1],
                sampled_over_counters=ticks[0] / ticks[1],
            )
        )
    return dict(
        passed=True,
        comparisons=rows,
        sources=[
            dict(
                bundle=str(p),
                results_sha256=hashlib.sha256(
                    (p / "results.json").read_bytes()
                ).hexdigest(),
            )
            for p in (full, lean)
        ],
        scope="Matched grouped GEMV SDK simulator maximum-local intervals. Both modes validate every final replica and phase/epoch/queue protocol. Counters mode omits phase stores/reads and local-compute timestamp sampling; no internal-phase observation, hardware, synchronized global latency or precision improvement claim.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sampled", type=Path)
    p.add_argument("counters", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    assert not a.output.exists(), "fresh evidence required"
    a.output.write_text(
        json.dumps(compare(a.sampled.resolve(), a.counters.resolve()), indent=2) + "\n"
    )
    print(a.output)
