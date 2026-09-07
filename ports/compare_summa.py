"""SUMMA uses the same frozen-audit comparison machinery as mesh GEMV."""

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
    if result["profile"] != "mesh_gemm.v1":
        raise ValueError("expected SUMMA artifacts")
    a.o.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
