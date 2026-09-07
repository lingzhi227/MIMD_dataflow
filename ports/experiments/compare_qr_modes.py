"""Compare preserved QR instrumentation runs without modifying either bundle.

Each run is re-audited by its frozen implementation in a temporary copy. The
native baseline is reused only after checking its recorded input and options.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_audit(root):
    with tempfile.TemporaryDirectory(prefix="qr-mode-audit-") as tmp:
        copy = Path(tmp) / "bundle"
        copy.mkdir()
        for p in root.iterdir():
            if p.is_file() and p.suffix not in (".core", ".log"):
                shutil.copy2(p, copy / p.name)
        shutil.copytree(root / "implementation", copy / "implementation")
        subprocess.run(
            [sys.executable, str(copy / "implementation/validate.py"), str(copy)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        return read(copy / "audit.json")


def durations(timing):
    ticks = [
        (
            sum(int(w[i + 3]) << (16 * i) for i in range(3))
            - sum(int(w[i]) << (16 * i) for i in range(3))
        )
        % (1 << 48)
        for row in timing
        for w in row
    ]
    require(all(0 < t < 1 << 32 for t in ticks), "timestamp bounds")
    return ticks


def compare(sampled, counters, baseline):
    sampled, counters, baseline = map(Path, (sampled, counters, baseline))
    sa, ca = frozen_audit(sampled), frozen_audit(counters)
    require(sa["passed"] and ca["passed"], "frozen audit failed")
    ss, cs = [read(p / "schedule.json") for p in (sampled, counters)]
    require(ss.get("instrumentation", "sampled") == "sampled", "sampled mode required")
    require(cs.get("instrumentation") == "counters", "counters mode required")
    for key in (
        "profile",
        "M",
        "N",
        "Nt",
        "rows",
        "cols",
        "epochs",
        "nodes",
        "resources",
    ):
        require(ss.get(key) == cs.get(key), "different schedule: " + key)
    require(ss["profile"] == "mesh_qr.v1", "QR required")
    for name in ("source.cpp", "batches.json", "runtime-options.json"):
        require(
            (sampled / name).read_bytes() == (counters / name).read_bytes(),
            "different " + name,
        )
    sr, cr = [read(p / "results.json") for p in (sampled, counters)]
    require(len(sr["cases"]) == len(cr["cases"]) == cs["epochs"], "epoch count")
    for a, b in zip(sr["cases"], cr["cases"]):
        require(a.keys() == b.keys(), "output names differ")
        for key in a:
            x, y = [np.asarray(v[key], dtype=np.float32) for v in (a, b)]
            require(
                x.shape == y.shape
                and np.array_equal(x.view(np.uint32), y.view(np.uint32)),
                "output bits differ",
            )
    for a, b in zip(sr["diagnostics"], cr["diagnostics"]):
        require(a["progress"] == b["progress"], "per-PE counts differ")
        require(np.count_nonzero(b["witnesses"]) == 0, "counter mode retained samples")
    bm = read(baseline / "manifest.json")
    for name, sha in bm["files"].items():
        require(digest(baseline / name) == sha, "baseline hash mismatch: " + name)
    require(
        bm["comparison_results_sha256"] == digest(sampled / "results.json"),
        "baseline belongs to another sampled run",
    )
    require(
        read(baseline / "runtime-options.json")
        == read(counters / "runtime-options.json"),
        "baseline runtime options differ",
    )
    require(read(baseline / "schedule.json") == ss, "baseline geometry differs")
    batches = read(counters / "batches.json")
    require(read(baseline / "input.json") == batches[0]["a"], "baseline input differs")
    require(read(baseline / "comparison.json")["passed"], "baseline failed")
    nt = durations(read(baseline / "results.json")["timing"])
    st = durations(sr["diagnostics"][0]["timing"])
    ct = durations(cr["diagnostics"][0]["timing"])
    return {
        "passed": True,
        "sampled": str(sampled.resolve()),
        "counters": str(counters.resolve()),
        "native_baseline": str(baseline.resolve()),
        "epochs": cs["epochs"],
        "all_output_f32_bits_equal": True,
        "all_per_pe_counts_equal": True,
        "counter_witness_storage_exact_zero": True,
        "sampled_max_local_cycles_epoch0": max(st),
        "counters_max_local_cycles_epoch0": max(ct),
        "native_max_local_cycles": max(nt),
        "sampled_over_native": max(st) / max(nt),
        "counters_over_native": max(ct) / max(nt),
        "sampled_per_pe_cycles_epoch0": st,
        "counters_per_pe_cycles_epoch0": ct,
        "native_per_pe_cycles": nt,
        "scope": "SDK simulator maximum local factor interval, epoch 0; excludes prepare and host I/O. Counts-only mode loses numerical rotation samples and retains zeroed witness storage/transfer. No hardware or synchronized global latency claim.",
        "evidence_sha256": {
            str(p.resolve()): digest(p)
            for root in (sampled, counters, baseline)
            for p in (root / "manifest.json", root / "results.json")
        },
        "comparison_script_sha256": digest(Path(__file__)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sampled")
    parser.add_argument("counters")
    parser.add_argument("baseline")
    parser.add_argument("-o", required=True)
    args = parser.parse_args()
    result = compare(args.sampled, args.counters, args.baseline)
    with Path(args.o).open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k.endswith("over_native") or k == "passed"
            }
        )
    )
