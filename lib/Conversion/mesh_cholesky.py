from source_tree import source_path, logical_name
"""Triangular right-looking Cholesky, preserving the SDK wavefront algorithm."""

import copy
import math
from pathlib import Path
from frontend import check
from float32 import f32


def factor_f32(values, n):
    check(len(values) == n * n, "Cholesky input size")
    a = list(map(f32, values))
    check(all(math.isfinite(x) for x in a), "Cholesky finite input")
    check(
        all(a[i * n + j] == a[j * n + i] for i in range(n) for j in range(i)),
        "Cholesky requires symmetric input",
    )
    for i in range(n):
        for j in range(i + 1, n):
            a[i * n + j] = 0.0
    for k in range(n):
        check(a[k * n + k] > 0, "Cholesky requires positive pivots")
        a[k * n + k] = f32(math.sqrt(a[k * n + k]))
        for i in range(k + 1, n):
            a[i * n + k] = f32(a[i * n + k] / a[k * n + k])
        for i in range(k + 1, n):
            for j in range(k + 1, i + 1):
                a[i * n + j] = f32(a[i * n + j] - f32(a[i * n + k] * a[j * n + k]))
    return a


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    check(
        [n["op"] for n in m["nodes"]] == ["input", "cholesky", "output"],
        "Cholesky profile requires input, factor, output",
    )
    a, op, out = m["nodes"]
    check(len({n["id"] for n in m["nodes"]}) == 3, "Cholesky unique ids")
    check(
        op["inputs"] == [a["id"]] and out["inputs"] == [op["id"]],
        "Cholesky dependencies",
    )
    check(
        not m["states"] and not any("place" in n for n in m["nodes"]),
        "Cholesky owns placement and state",
    )
    check(not any("dataflow" in n for n in (a, out)), "Cholesky annotation target")
    shape = a["shape"]
    check(
        len(shape) == 2
        and all(type(v) is int for v in shape)
        and 2 <= shape[0] == shape[1] <= 256
        and op["shape"] == shape,
        "Cholesky square matrix dimension 2..256",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 32767,
        "Cholesky execution bounds",
    )
    d = op["dataflow"]
    check(
        d
        == dict(
            rows=d["rows"],
            cols=d["cols"],
            triangle="lower",
            update="right_looking",
            fp="relaxed",
            compute="vector",
        ),
        "Cholesky triangular right-looking vector policy",
    )
    check(
        type(d["rows"]) is int
        and type(d["cols"]) is int
        and 2 <= d["rows"] == d["cols"] <= 8,
        "Cholesky square PE mesh 2..8",
    )
    out["shape"] = shape[:]
    m.update(profile="mesh_cholesky.v1", epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    check(partitions == 1, "Cholesky owns partitioning")
    a, op, out = m["nodes"]
    n, p = a["shape"][0], op["dataflow"]["rows"]
    check(n % p == 0, "Cholesky divisible tile dimension")
    nt = n // p
    memory = {
        "tile": 4 * nt * nt,
        "panels": 8 * nt,
        "diagnostics": 28 + 8 * n,
        "sdk_control_reserve": 8192,
    }
    check(
        sum(memory.values()) <= 48 * 1024 and nt * (nt + 1) <= 32767,
        "Cholesky memory/DSD bounds",
    )
    return {
        "profile": "mesh_cholesky.v1",
        "N": n,
        "P": p,
        "Nt": nt,
        "epochs": m["epochs"],
        "tile_order": "row-major",
        "nodes": [
            {
                "id": f"p{x}_{y}",
                "tile": [x, y],
                "place": [4 + x, 1 + y],
                "active": x <= y,
            }
            for y in range(p)
            for x in range(p)
        ],
        "memory_per_pe": memory,
        "stages": [
            "global prepare barrier: reset DSD, task, switch position",
            "pivot scale and column multicast",
            "left fringe scale and row multicast",
            "joined row/column receive and resident rank-1 update",
            "retire left fringe with switch-advance control wavelets",
        ],
        "resources": {
            "colors": [0, 1],
            "local_tasks": [17, 18],
            "column_input_output_queues": [2, 3],
            "row_input_output_queues": [4, 5],
            "memcpy_queues": [0, 1],
        },
        "numeric_policy": "SPD input; SDK invsqrt_f32 and DSD FMA; fixed factor tolerance plus relative LLT residual",
        "lifecycle": "same runtime, prepare barrier before each factorization",
    }


def generate(s, dest):
    runtime = source_path("runtime")
    for source, target in (("pe", "pe"), ("layout", "layout"), ("launch", "launch")):
        (Path(dest) / (target + ".csl")).write_text(
            (runtime / ("mesh_cholesky_" + source + ".csl")).read_text()
        )
