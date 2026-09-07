"""Cannon uses the same frozen-audit comparison machinery as mesh GEMV."""

import argparse
import json
from pathlib import Path
from compare_mesh_gemv import compare

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("vector", type=Path)
    p.add_argument("scalar", type=Path)
    p.add_argument("-o", type=Path, required=True)
    a = p.parse_args()
    result = compare(a.vector, a.scalar)
    if result["profile"] != "mesh_cannon.v1":
        raise ValueError("expected Cannon artifacts")
    import numpy as np

    audits = [
        json.loads((root / "audit.json").read_text()) for root in (a.vector, a.scalar)
    ]
    result["scalar_over_vector_max_local_interval"] = [
        float(np.max(st) / np.max(vt))
        for vt, st in zip(
            audits[0]["total_cycles_per_epoch_pe"],
            audits[1]["total_cycles_per_epoch_pe"],
        )
    ]
    result["resident_interval_scope"] = (
        "Maximum PE-local main interval, includes reset, ring communication and diagnostics; excludes host transfers. Not synchronized global latency."
    )
    a.o.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
