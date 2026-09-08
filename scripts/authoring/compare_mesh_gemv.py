"""Controlled compute-lowering comparison on identical GEMV topology and inputs."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse
import json
import subprocess
import sys
import hashlib
from pathlib import Path
import numpy as np


def read(root, name):
    return json.loads((root / name).read_text())


def compare(vector, scalar):
    for root in (vector, scalar):
        subprocess.run(
            [sys.executable, str(root / "implementation/validate.py"), str(root)],
            capture_output=True,
            text=True,
            check=True,
        )
    vs, ss = (read(p, "schedule.json") for p in (vector, scalar))
    if vs["compute"] != "vector" or ss["compute"] != "scalar":
        raise ValueError("expected vector then scalar")
    if {k: v for k, v in vs.items() if k != "compute"} != {
        k: v for k, v in ss.items() if k != "compute"
    }:
        raise ValueError("different schedule")
    if read(vector, "batches.json") != read(scalar, "batches.json"):
        raise ValueError("different inputs")
    runtime_options = [
        (
            read(p, "runtime-options.json")
            if (p / "runtime-options.json").exists()
            else None
        )
        for p in (vector, scalar)
    ]
    if runtime_options[0] != runtime_options[1]:
        raise ValueError("different SDK runtime instrumentation")
    audits = [read(p, "audit.json") for p in (vector, scalar)]
    if not all(a["passed"] for a in audits):
        raise ValueError("audit failed")
    vr, sr = (read(p, "results.json") for p in (vector, scalar))
    pair_fixed_accuracy = True
    for va, sa in zip(vr["cases"], sr["cases"]):
        if va.keys() != sa.keys():
            raise ValueError("output ports differ")
        for k in va:
            pair_fixed_accuracy = pair_fixed_accuracy and bool(
                np.allclose(va[k], sa[k], rtol=3e-5, atol=3e-6)
            )
            if vs["profile"] not in ("mesh_gemm.v1", "mesh_cannon.v1"):
                np.testing.assert_allclose(va[k], sa[k], rtol=3e-5, atol=3e-6)
    vc, sc = (a["compute_cycles_median"] for a in audits)
    totals = [read(p, "sim_stats.json")["cycle_count"] for p in (vector, scalar)]
    return {
        "vector": str(vector),
        "scalar": str(scalar),
        "same_schedule_except_compute": True,
        "same_inputs": True,
        "same_runtime_options": True,
        "runtime_options": runtime_options[0],
        "result_sha256": {
            str(p): hashlib.sha256((p / "results.json").read_bytes()).hexdigest()
            for p in (vector, scalar)
        },
        "both_numerically_audited": True,
        "profile": vs["profile"],
        "pair_fixed_accuracy_passed": pair_fixed_accuracy,
        "individual_fixed_accuracy_passed": [
            a.get("fixed_accuracy_passed") for a in audits
        ],
        "vector_local_cycles_median": vc,
        "scalar_local_cycles_median": sc,
        "local_compute_ratio": sc / vc,
        "vector_total_sim_cycles": totals[0],
        "scalar_total_sim_cycles": totals[1],
        "total_sim_ratio": totals[1] / totals[0],
        "scope": "Use the individual frozen audits for each numerical acceptance contract. SDK simulator: same matrix, PE mesh, library communication and four input epochs. Local ratio excludes communication/host I/O; total includes transfers and diagnostics. Relaxed FP allows FMA/reassociation. Not hardware speedup or parity with tuned vendor BLAS.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("vector", type=Path)
    p.add_argument("scalar", type=Path)
    p.add_argument("-o", type=Path, required=True)
    args = p.parse_args()
    result = compare(args.vector, args.scalar)
    args.o.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
