"""No-pivot distributed blocked LU, using the Matrix algorithms dataflow."""

import copy
import math
from pathlib import Path
from frontend import check
from float32 import f32


def factor(values, n, single=True):
    cast = f32 if single else float
    a = list(map(cast, values))
    check(len(a) == n * n and all(math.isfinite(x) for x in a), "LU finite input shape")
    # Conservative explicit first profile domain; never silently adjust the diagonal.
    check(
        all(
            abs(a[i * n + i]) > math.fsum(abs(a[i * n + j]) for j in range(n) if i != j)
            for i in range(n)
        ),
        "LU profile requires strict row diagonal dominance",
    )
    for k in range(n):
        check(
            math.isfinite(a[k * n + k]) and a[k * n + k] != 0, "LU zero/nonfinite pivot"
        )
        for i in range(k + 1, n):
            a[i * n + k] = cast(a[i * n + k] / a[k * n + k])
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a[i * n + j] = cast(a[i * n + j] - cast(a[i * n + k] * a[k * n + j]))
    return a


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    check(
        [n["op"] for n in m["nodes"]] == ["input", "lu_no_pivot", "output"],
        "LU requires input, packed factor, output",
    )
    a, op, out = m["nodes"]
    check(
        len({n["id"] for n in m["nodes"]}) == 3
        and op["inputs"] == [a["id"]]
        and out["inputs"] == [op["id"]],
        "LU dependencies",
    )
    check(
        not m["states"]
        and not any("place" in n for n in m["nodes"])
        and not any("dataflow" in n for n in (a, out)),
        "LU placement ownership",
    )
    shape = a["shape"]
    check(
        len(shape) == 2
        and all(type(v) is int for v in shape)
        and 2 <= shape[0] == shape[1] <= 256
        and op["shape"] == shape,
        "LU square matrix 2..256",
    )
    d = op["dataflow"]
    check(
        d
        == dict(
            rows=d["rows"],
            cols=d["cols"],
            pivot="none",
            update="blocked",
            fp="relaxed",
            compute="vector",
        ),
        "LU blocked no-pivot policy",
    )
    check(
        type(d["rows"]) is int
        and type(d["cols"]) is int
        and 2 <= d["rows"] == d["cols"] <= 8,
        "LU square mesh 2..8",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 32767,
        "LU execution bounds",
    )
    out["shape"] = shape[:]
    m.update(profile="mesh_lu.v1", epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    check(partitions == 1, "LU owns partitioning")
    n, p = m["nodes"][0]["shape"][0], m["nodes"][1]["dataflow"]["rows"]
    check(n % p == 0 and n // p >= 2, "LU requires divisible square tiles of width >=2")
    nt = n // p
    memory = {
        "tile": 4 * nt * nt,
        "scratch_panels": 12 * nt,
        "diagnostics": 28 + 8 * n,
        "sdk_control_reserve": 8192,
    }
    check(
        sum(memory.values()) <= 48 * 1024 and nt * (nt + 1) <= 32767,
        "LU memory/DSD bounds",
    )
    return {
        "profile": "mesh_lu.v1",
        "N": n,
        "P": p,
        "Nt": nt,
        "epochs": m["epochs"],
        "tile_order": "row-major",
        "memory_per_pe": memory,
        "nodes": [
            {"id": f"p{x}_{y}", "tile": [x, y], "place": [4 + x, 1 + y]}
            for y in range(p)
            for x in range(p)
        ],
        "stages": [
            "prepare barrier resets queues, DSDs and receive routes",
            "consume complete preceding block-pivot prefix",
            "switch completed-prefix rows/columns to local injection",
            "diagonal local elimination; division and row-signal data tasks",
            "resident vector rank-1 updates and retire",
        ],
        "resources": {
            "colors": [0, 1, 2, 3],
            "local_tasks": [10],
            "data_task_input_queues": [4, 5],
            "fabric_input_queues": [2, 3],
            "output_queues": [4, 5, 6, 7],
            "memcpy_queues": [0, 1],
        },
        "domain": "strict row diagonal dominance; nonzero pivots; no host diagonal adjustment",
        "output": "packed LU: strict lower stores L multipliers, implicit unit L diagonal; upper stores U",
        "lifecycle": "one runtime; global prepare barrier before each call",
    }


def generate(s, dest):
    runtime = Path(__file__).parent / "runtime"
    for name in ("pe", "layout"):
        (Path(dest) / (name + ".csl")).write_text(
            (runtime / ("mesh_lu_" + name + ".csl")).read_text()
        )
