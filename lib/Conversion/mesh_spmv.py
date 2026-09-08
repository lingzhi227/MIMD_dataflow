from source_tree import source_path, logical_name
"""Typed canonical CSC SpMV lowered to the SDK's two-direction train schedule."""

import copy
from pathlib import Path
from frontend import check
from float32 import f32
from sparse_storage import CSC, Capacity, geometry, partition

TEMPLATES = {
    "u16_transport.csl": "u16_transport.csl",
    "layout.csl": "spmv_layout.csl",
    "kernel.csl": "spmv_kernel.csl",
    "spmv_pe.csl": "spmv_pe.csl",
    "spmv_routes.csl": "spmv_routes.csl",
}


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    check(
        [n["op"] for n in m["nodes"]]
        == ["input", "index_input", "index_input", "input", "spmv_csc", "output"],
        "SpMV requires values, u32 row indices, u32 offsets, x, spmv, output",
    )
    values, rows, offsets, x, op, out = m["nodes"]
    check(
        len({n["id"] for n in m["nodes"]}) == 6
        and op["inputs"] == [n["id"] for n in m["nodes"][:4]]
        and out["inputs"] == [op["id"]],
        "SpMV dependencies",
    )
    check(
        not m["states"]
        and not any("place" in n for n in m["nodes"])
        and not any("dataflow" in n for n in m["nodes"] if n is not op),
        "SpMV owns layout",
    )
    check(len({n["host"] for n in m["nodes"][:4]}) == 4, "unique sparse input ports")
    for n in m["nodes"][:5]:
        check(
            len(n["shape"]) == 2
            and all(type(v) is int and v > 0 for v in n["shape"])
            and n["shape"][1] == 1,
            "sparse vector shape",
        )
    M, N, nnz = op["shape"][0], x["shape"][0], values["shape"][0]
    check(
        max(M, N) <= 65534
        and nnz <= 262144
        and rows["shape"] == [nnz, 1]
        and offsets["shape"] == [N + 1, 1],
        "CSC extents",
    )
    d = op["dataflow"]
    check(
        d
        == dict(
            rows=d["rows"],
            cols=d["cols"],
            storage="csc",
            exchange="trains",
            reduce="sparse_rows",
            nnz_per_pe=d["nnz_per_pe"],
            cols_per_pe=d["cols_per_pe"],
            rows_per_pe=d["rows_per_pe"],
            fp="relaxed",
        ),
        "SpMV dataflow policy",
    )
    check(4 <= d["rows"] <= 8 and 2 <= d["cols"] <= 8, "bounded sparse PE grid")
    cap = Capacity(d["nnz_per_pe"], d["cols_per_pe"], d["rows_per_pe"])
    g = geometry(M, N, d["rows"], d["cols"])
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 32767,
        "epoch/bound contract",
    )
    for n in m["nodes"]:
        n["dtype"] = "u32" if n["op"] == "index_input" else "f32"
    out["shape"] = [M, 1]
    m.update(
        profile="mesh_spmv.v1",
        epochs=epochs,
        input_bound=bound,
        sparse_capacity=cap.__dict__,
        sparse_geometry=g,
    )
    return m


def matrix(m, batch):
    v, r, p, x, op, out = m["nodes"]
    check(set(batch) == {n["host"] for n in m["nodes"][:4]}, "sparse input ports")
    for node in m["nodes"][:4]:
        check(len(batch[node["host"]]) == node["shape"][0], "sparse input extent")
    for node in (v, x):
        check(
            all(
                type(a) in (int, float) and abs(a) <= m["input_bound"]
                for a in batch[node["host"]]
            ),
            "sparse f32 input bound",
        )
    a = CSC(
        op["shape"][0],
        x["shape"][0],
        batch[p["host"]],
        batch[r["host"]],
        batch[v["host"]],
    )
    partition(
        a,
        m["sparse_geometry"]["prows"],
        m["sparse_geometry"]["pcols"],
        Capacity(**m["sparse_capacity"]),
    )
    from sparse_storage import finite_f32

    for z in batch[x["host"]]:
        finite_f32(z)
    return a


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "sparse epoch count")
    results, history = [], {n["id"]: [] for n in m["nodes"]}
    for batch in batches:
        a = matrix(m, batch)
        x = batch[m["nodes"][3]["host"]]
        y = [0.0] * a.rows
        for row, col, value in a.entries():
            y[row] = f32(y[row] + f32(value * x[col]))
        results.append({m["nodes"][-1]["host"]: y})
    return results, history


def resource_contract():
    result = dict(
        colors=list(range(1, 7)),
        local_tasks=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26],
        spmv_input_queues=[4, 2, 6, 7],
        spmv_output_queues=[2, 3, 4, 5],
        memcpy_input_queues=[0, 1],
        memcpy_output_queues=[0, 1],
        dest_dsr=[1, 4, 5, 6, 2, 3],
        src1_dsr=[4, 1, 6, 7, 2, 3],
        phase_microthreads={
            "north_south": {"receive": [4, 2], "send": [0, 3]},
            "east_west": {"receive": [6, 7], "send": [0, 3]},
        },
        output_color_phases={
            "north_south": ["north_train", "south_train"],
            "east_west": ["tx_west_train", "tx_east_train"],
        },
        lifetime="Command stream blocked during SpMV; UT0 borrowed after host transfers and released after joined sends before callback. Each phase has fixed-color output queues; no runtime output recoloring.",
        timing="SDK local timestamp intervals; no auxiliary clock allreduce linked",
    )
    validate_resources(result)
    return result


def validate_resources(r):
    def unique(values, upper, name):
        check(
            all(type(v) is int and 0 <= v < upper for v in values)
            and len(set(values)) == len(values),
            "resource conflict: " + name,
        )

    unique(r["colors"], 24, "colors")
    check(not set(r["colors"]) & {21, 22, 23}, "SDK colors")
    unique(r["local_tasks"], 31, "local tasks")
    check(all(v >= 8 for v in r["local_tasks"]), "WSE3 local task range8..30")
    check(
        not set(r["local_tasks"]) & {21, 22, 23, 27, 28, 29, 30, 31}, "SDK local tasks"
    )
    unique(r["spmv_input_queues"] + r["memcpy_input_queues"], 8, "input queues")
    unique(r["spmv_output_queues"] + r["memcpy_output_queues"], 8, "output queues")
    unique(r["dest_dsr"], 8, "destination DSRs")
    unique(r["src1_dsr"], 8, "source1 DSRs")
    for phase, owners in r["phase_microthreads"].items():
        unique(owners["receive"] + owners["send"], 8, "microthreads " + phase)
        check(1 not in owners["receive"] + owners["send"], "command microthread")


def plan(m, partitions=1):
    check(partitions == 1, "SpMV uses explicit PE geometry")
    g, c = m["sparse_geometry"], m["sparse_capacity"]
    empty = CSC(
        m["nodes"][4]["shape"][0],
        m["nodes"][3]["shape"][0],
        [0] * (m["nodes"][3]["shape"][0] + 1),
        [],
        [],
    )
    storage = partition(empty, g["prows"], g["pcols"], Capacity(**c))
    return dict(
        profile="mesh_spmv.v1",
        M=empty.rows,
        N=empty.cols,
        epochs=m["epochs"],
        rows=g["prows"],
        cols=g["pcols"],
        geometry=g,
        capacity=c,
        extents=storage["extents"],
        estimated_bytes=storage["estimated_bytes"],
        nodes=[
            dict(id=f"p{x}_{y}", place=[x + 4, y + 1])
            for y in range(g["prows"])
            for x in range(g["pcols"])
        ],
        stages=[
            "canonical CSC validation",
            "capacity-checked compact sparse partition",
            "north/south vector trains and local sparse products",
            "east/west sparse row merge and reduction",
            "distributed dense y",
        ],
        resources=resource_contract(),
        domain="canonical sorted duplicate-free CSC; explicit zeros retained; u32 host indices checked before u16 tile narrowing",
    )


def generate(schedule, dest):
    dest = Path(dest)
    for name, template in TEMPLATES.items():
        (dest / name).write_text(
            (source_path("runtime") / template).read_text()
        )
