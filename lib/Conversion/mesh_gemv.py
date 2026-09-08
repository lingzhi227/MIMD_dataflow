from source_tree import source_path, logical_name
"""Checked two-dimensional matvec dataflow over SDK collectives_2d."""

import copy
from pathlib import Path
from frontend import check


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    check(
        [n["op"] for n in m["nodes"]] == ["input", "input", "matmul", "output"],
        "mesh matvec requires A, x, matmul, output",
    )
    a, x, op, out = m["nodes"]
    d = op["dataflow"]
    rows, cols = a["shape"]
    check(
        all(type(v) is int and v > 0 for v in (rows, cols, d["rows"], d["cols"])),
        "mesh extents",
    )
    check(2 <= d["rows"] <= 8 and 2 <= d["cols"] <= 8, "mesh rows/cols 2..8")
    check(rows <= 512 and cols <= 512 and rows * cols <= 262144, "mesh matrix bounds")
    check(
        rows % d["rows"] == 0 and cols % d["cols"] == 0,
        "mesh requires evenly divisible blocks",
    )
    check(x["shape"] == [cols, 1] and op["shape"] == [rows, 1], "mesh GEMV shapes")
    check(
        op["inputs"] == [a["id"], x["id"]] and out["inputs"] == [op["id"]],
        "mesh dependencies",
    )
    check(
        d["fp"] == "relaxed" and d["broadcast"] == "columns" and d["reduce"] == "rows",
        "mesh numerical/dataflow policy",
    )
    check(d["compute"] in ("vector", "scalar"), "mesh compute policy")
    check(
        not m["states"] and not any("place" in n for n in m["nodes"]),
        "mesh owns placement",
    )
    check(
        a["host"] != x["host"] and len({n["id"] for n in m["nodes"]}) == 4,
        "mesh unique ports/ids",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 32767,
        "mesh execution bounds",
    )
    out["shape"] = [rows, 1]
    for n in m["nodes"]:
        n["interval"] = None
    m.update(profile="mesh_gemv.v1", epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    check(partitions == 1, "mesh dataflow owns partitioning")
    a, x, op, out = m["nodes"]
    d = op["dataflow"]
    rows, cols = a["shape"]
    mt, nt = rows // d["rows"], cols // d["cols"]
    memory = {
        "matrix": mt * nt * 4,
        "root_vector": cols * 4,
        "vector_partition": nt * 4,
        "partial_and_reduced": 2 * mt * 4,
        "gather_buffer": rows * 4,
        "sdk_control_reserve": 8192,
    }
    check(sum(memory.values()) <= 48 * 1024, "mesh PE memory budget")
    check(mt * nt <= 32767, "mesh matrix DSD offset i16")
    return {
        "profile": "mesh_gemv.v1",
        "matrix_rows": rows,
        "matrix_cols": cols,
        "kernel_rows": d["rows"],
        "kernel_cols": d["cols"],
        "Mt": mt,
        "Nt": nt,
        "compute": d["compute"],
        "epochs": m["epochs"],
        "nodes": [],
        "memory_per_pe": memory,
        "stages": [
            "host distributes A blocks; root receives x",
            "scatter x across top row",
            "broadcast x partitions down columns",
            "local block matvec",
            "reduce partials across rows",
            "gather row results down right column",
        ],
        "numeric_policy": "explicit relaxed: local FMA allowed, partitioned row reduction",
        "library": "SDK 2.10.1 <collectives_2d/pe>",
        "resources": {
            "colors": [0, 1, 4, 5],
            "local_tasks": list(range(9, 18)),
            "library_queue_dsr_policy": "SDK collectives_2d defaults; x/y modules own separate dimensions",
        },
    }


def generate(s, dest):
    runtime = source_path("runtime")
    body = (
        """for (@range(i16,Nt)) |j| {
      const column=@increment_dsd_offset(dsd_A_tile,j,f32);
      @fmacs(dsd_local_prod,dsd_local_prod,column,x_tile[j]);
    }"""
        if s["compute"] == "vector"
        else """for (@range(u16,Nt)) |j| {
      for (@range(u16,Mt)) |i| { local_prod[i] += A_tile[i*Nt+j]*x_tile[j]; }
    }"""
    )
    pe = (runtime / "mesh_gemv_pe.csl").read_text()
    check(pe.count("// HLS_COMPUTE_BODY") == 1, "mesh runtime placeholder")
    (Path(dest) / "pe.csl").write_text(pe.replace("// HLS_COMPUTE_BODY", body))
    (Path(dest) / "layout.csl").write_text(
        (runtime / "mesh_gemv_layout.csl").read_text()
    )
