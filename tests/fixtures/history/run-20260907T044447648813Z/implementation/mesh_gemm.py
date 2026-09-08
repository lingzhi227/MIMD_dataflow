"""SUMMA matmul: orthogonal broadcasts, joined completion, resident accumulation."""

from pathlib import Path
from frontend import check
from mesh_common import verify_matmul


def verify(module, epochs, bound):
    m = verify_matmul(module, epochs, bound)
    d = m["nodes"][2]["dataflow"]
    check(
        type(d["rows"]) is int and d["rows"] == d["cols"] and 2 <= d["rows"] <= 8,
        "SUMMA requires square PE grid 2..8",
    )
    check(
        d["broadcast"] == "rows_columns" and d["reduce"] == "local",
        "SUMMA dataflow policy",
    )
    m.update(profile="mesh_gemm.v1")
    plan(m)
    return m


def plan(m, partitions=1):
    check(partitions == 1, "SUMMA owns reduction partitioning")
    a, b, op, out = m["nodes"]
    rows, k = a["shape"]
    cols = b["shape"][1]
    d = op["dataflow"]
    p = d["rows"]
    check(
        all(v % p == 0 for v in (rows, k, cols)),
        "SUMMA requires evenly divisible blocks",
    )
    mt, kt, nt = rows // p, k // p, cols // p
    memory = {
        "resident_A_B_C": 4 * (mt * kt + kt * nt + mt * nt),
        "broadcast_buffers": 4 * (mt * kt + kt * nt),
        "round_history": 4 * p * mt * nt,
        "timestamps": 12 * p,
        "sdk_control_reserve": 8192,
    }
    check(sum(memory.values()) <= 48 * 1024, "SUMMA PE memory budget")
    check(max(mt * kt, kt * nt, p * mt * nt) <= 32767, "SUMMA DSD signed offset bound")
    return {
        "profile": "mesh_gemm.v1",
        "matrix_rows": rows,
        "matrix_k": k,
        "matrix_cols": cols,
        "P": p,
        "Mt": mt,
        "Kt": kt,
        "Nt": nt,
        "epochs": m["epochs"],
        "compute": d["compute"],
        "nodes": [
            {"id": f"p{x}_{y}", "tile": [x, y], "place": [4 + x, 1 + y]}
            for y in range(p)
            for x in range(p)
        ],
        "memory_per_pe": memory,
        "tile_order": "column-major",
        "stages": [
            {"id": "broadcast_A", "axis": "row", "root": "round"},
            {"id": "broadcast_B", "axis": "column", "root": "round"},
            {
                "id": "compute",
                "after": ["broadcast_A", "broadcast_B"],
                "operation": "C += A_panel * B_panel",
            },
        ],
        "rounds": p,
        "numeric_policy": "explicit relaxed FMA; increasing global K order",
        "validation_policy": "componentwise-f32-dot-v1; gamma_(2K+1)*abs(A)@abs(B)",
        "resources": {
            "colors": [0, 1, 4, 5],
            "local_tasks": list(range(8, 16)),
            "x_queues": [2, 4],
            "y_queues": [3, 5],
            "x_dsr_dest_src0_src1": 1,
            "y_dsr_dest_src0_src1": 2,
            "memcpy_queues": [0, 1],
            "completion_join": "x activates blocked compute; y unblocks it; compute blocks before next round",
        },
    }


def generate(s, dest):
    runtime = Path(__file__).parent / "runtime"
    body = (
        (runtime / "mesh_gemm_vector.csl").read_text()
        if s["compute"] == "vector"
        else """
    for (@range(i16,Kt)) |k| {
      for (@range(i16,Nt)) |j| {
        for (@range(i16,Mt)) |i| { C_tile[j*Mt+i] += Ap.*[k*Mt+i]*Bp.*[j*Kt+k]; }
      }
    }
    """
    )
    pe = (runtime / "mesh_gemm_pe.csl").read_text()
    check(pe.count("// HLS_COMPUTE_BODY") == 1, "SUMMA compute marker")
    (Path(dest) / "pe.csl").write_text(pe.replace("// HLS_COMPUTE_BODY", body))
    (Path(dest) / "layout.csl").write_text(
        (runtime / "mesh_gemm_layout.csl").read_text()
    )
