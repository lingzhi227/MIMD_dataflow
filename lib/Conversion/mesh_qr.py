from source_tree import source_path, logical_name
"""Blocked neighbor-Givens QR returning R; floating row signs are unspecified."""

import copy
import math
from pathlib import Path
from frontend import check
from float32 import f32


def factor(values, rows, cols):
    a = list(map(f32, values))
    check(len(a) == rows * cols, "QR input shape")
    import numpy as np

    check(
        float(np.linalg.cond(np.asarray(a).reshape(rows, cols))) <= 10000,
        "QR condition limit 10000 before SDK",
    )
    scale = max(map(abs, a))
    check(scale > 0, "QR nonzero full-rank input required")
    for col in range(cols):
        for row in range(rows - 1, col, -1):
            x, y = a[(row - 1) * cols + col], a[row * cols + col]
            if y == 0:
                c, s = 1.0, 0.0
            elif abs(y) > abs(x):
                t = f32(-x / y)
                s = f32(1 / f32(math.sqrt(f32(1 + f32(t * t)))))
                c = f32(s * t)
            else:
                t = f32(-y / x)
                c = f32(1 / f32(math.sqrt(f32(1 + f32(t * t)))))
                s = f32(c * t)
            for j in range(col, cols):
                upper, lower = a[(row - 1) * cols + j], a[row * cols + j]
                a[(row - 1) * cols + j] = f32(f32(c * upper) - f32(s * lower))
                a[row * cols + j] = f32(f32(s * upper) + f32(c * lower))
        check(abs(a[col * cols + col]) > 2**-17 * scale, "QR rank/pivot margin")
    return a


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    check(
        [n["op"] for n in m["nodes"]] == ["input", "qr_r", "output"],
        "QR input, R, output profile",
    )
    a, op, out = m["nodes"]
    check(
        len({n["id"] for n in m["nodes"]}) == 3
        and op["inputs"] == [a["id"]]
        and out["inputs"] == [op["id"]],
        "QR dependencies",
    )
    check(
        not m["states"]
        and not any("place" in n for n in m["nodes"])
        and not any("dataflow" in n for n in (a, out)),
        "QR owns placement",
    )
    shape = a["shape"]
    check(
        len(shape) == 2
        and all(type(v) is int for v in shape)
        and 2 <= shape[1] <= shape[0] <= 256
        and op["shape"] == shape,
        "QR tall/square shape <=256",
    )
    d = op["dataflow"]
    check(
        d
        == dict(
            rows=d["rows"],
            cols=d["cols"],
            exchange="neighbors",
            rotation="givens",
            fp="relaxed",
            compute="vector",
        ),
        "QR neighbor Givens policy",
    )
    check(
        type(d["rows"]) is int
        and type(d["cols"]) is int
        and 2 <= d["rows"] <= 8
        and 1 <= d["cols"] <= d["rows"],
        "QR mesh dimensions",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 32767,
        "QR execution bounds",
    )
    out["shape"] = shape[:]
    m.update(profile="mesh_qr.v1", epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    check(partitions == 1, "QR owns partitioning")
    rows, cols = m["nodes"][0]["shape"]
    d = m["nodes"][1]["dataflow"]
    py, px = d["rows"], d["cols"]
    check(
        rows % py == cols % px == 0 and rows // py == cols // px and rows // py >= 2,
        "QR requires divisible square local tiles >=2",
    )
    nt = rows // py
    mode = m.get("instrumentation", "sampled")
    check(mode in ("sampled", "counters"), "QR instrumentation mode")
    memory = {
        "tile": 4 * nt * nt,
        "two_row_buffers": 8 * nt,
        "rotation_witnesses": 512,
        "timing_progress": 28,
        "sdk_control_reserve": 8192,
    }
    check(
        sum(memory.values()) <= 48 * 1024
        and nt * (nt + 1) <= 32767
        and rows * cols < 65536,
        "QR memory/DSD/rotation counter bounds",
    )
    from qr_schedule import rotations

    return {
        "instrumentation": mode,
        "expected_rotation_counts": [
            [len(rotations(py, px, nt, x, y)) for x in range(px)] for y in range(py)
        ],
        "profile": "mesh_qr.v1",
        "M": rows,
        "N": cols,
        "Nt": nt,
        "rows": py,
        "cols": px,
        "epochs": m["epochs"],
        "tile_order": "row-major",
        "memory_per_pe": memory,
        "nodes": [
            {"id": f"p{x}_{y}", "tile": [x, y], "place": [4 + x, 1 + y]}
            for y in range(py)
            for x in range(px)
        ],
        "stages": [
            "prepare barrier restores DSDs and coefficient receive routes",
            "block-local adjacent-row Givens annihilation",
            "paired north/south row exchange",
            "broadcast sine/cosine east to remaining column blocks",
            "retire upper-trapezoidal R blocks",
        ],
        "resources": {
            "colors": [0, 2, 3, 4, 5],
            "input_queues": [2, 3, 4],
            "output_queues": [5, 6, 7],
            "memcpy_queues": [0, 1],
            "neighbor_colors": "direction-specific, alternating row parity",
        },
        "domain": "tall full column rank; native pivot margin >2^-17*maxabs(A); validation cond2(A)<=10000",
        "output": "R only, MxN upper trapezoid, row signs unspecified; Q not emitted",
        "rotation_sampling": (
            "none; exact per-PE counters"
            if mode == "counters"
            else "first 7 plus rolling 9 slots sampled every 16 rotations"
        ),
    }


def generate(s, dest):
    runtime = source_path("runtime")
    for name in ("pe", "layout"):
        code = (runtime / ("mesh_qr_" + name + ".csl")).read_text()
        if name == "pe" and s["instrumentation"] == "counters":
            import re

            code, count = re.subn(
                r"(?m)^(\s*)witness_begin\([^;\n]*\);", r"\1rotation_count += 1;", code
            )
            check(count == 3, "QR three rotation instrumentation sites")
            code = re.sub(r"(?m)^\s*witness_value\([^;\n]*\);", "", code)
        (Path(dest) / (name + ".csl")).write_text(code)
