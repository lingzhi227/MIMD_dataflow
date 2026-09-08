from source_tree import source_path, logical_name
"""Distributed scalar reductions with contiguous vector ownership and SDK collectives."""

import copy, math
from pathlib import Path
from frontend import check
from float32 import f32

TEMPLATES = {
    "layout.csl": "reduction_layout.csl",
    "pe.csl": "reduction_pe.csl",
    "scalar_allreduce.csl": "scalar_allreduce.csl",
    "blas.csl": "sdk_blas.csl",
}


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    ns = m["nodes"]
    op = ns[-2]
    kind = op["op"]
    arity = {"dot": 2, "nrm2": 1}.get(kind)
    check(
        arity is not None
        and len(ns) == arity + 2
        and [n["op"] for n in ns] == ["input"] * arity + [kind, "output"],
        "reduction graph",
    )
    inputs = ns[:arity]
    out = ns[-1]
    N = inputs[0]["shape"][0]
    d = op["dataflow"]
    check(
        set(d) == {"rows", "cols", "partition", "reduce", "result", "fp", "compute"},
        "reduction policy keys",
    )
    check(
        d["partition"] == "contiguous"
        and d["reduce"] == "row_column"
        and d["result"] == "replicated"
        and d["fp"] == "relaxed"
        and d["compute"] == "map",
        "reduction policy",
    )
    check(
        all(type(d[k]) is int and 2 <= d[k] <= 8 for k in ("rows", "cols")),
        "reduction mesh2..8",
    )
    check(
        type(N) is int
        and 1 <= N <= 262144
        and all(n["shape"] == [N, 1] for n in inputs)
        and op["shape"] == [1, 1],
        "reduction shapes",
    )
    check(
        op["inputs"] == [n["id"] for n in inputs]
        and out["inputs"] == [op["id"]]
        and len({n["id"] for n in ns}) == len(ns),
        "reduction dependencies",
    )
    check(
        len({n["host"] for n in inputs}) == arity
        and not m["states"]
        and not any("place" in n for n in ns)
        and not any("dataflow" in n for n in ns if n is not op),
        "reduction owns layout",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 32767,
        "reduction execution bounds",
    )
    out["shape"] = [1, 1]
    for n in ns:
        n["interval"] = None
        n["dtype"] = "f32"
    m.update(profile="mesh_reduction.v1", epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    check(partitions == 1, "reduction owns partitioning")
    op = m["nodes"][-2]
    d = op["dataflow"]
    N = m["nodes"][0]["shape"][0]
    h, w = d["rows"], d["cols"]
    local = (N + h * w - 1) // (h * w)
    check(local <= 4096, "local reduction length1..4096")
    memory = dict(
        vectors=8 * local,
        scalar_collective=4 * (h + w + 16),
        diagnostics=64,
        sdk_control_reserve=8192,
    )
    check(sum(memory.values()) <= 48 * 1024, "reduction PE memory")
    return dict(
        profile="mesh_reduction.v1",
        operation=op["op"],
        N=N,
        rows=h,
        cols=w,
        local_length=local,
        epochs=m["epochs"],
        nodes=[],
        memory_per_pe=memory,
        stages=[
            "host distributes contiguous vectors with zero padding",
            "SDK blas map local reduction",
            "SDK row reduction/gather, then column reduction/gather",
            "column broadcast followed by row broadcast",
            "all PEs retain the scalar",
        ],
        resources=dict(
            colors=[0, 1, 4, 5],
            local_tasks=[9, 10, 14, 15, 16, 17],
            sdk_collective_input_queues=[2, 4, 3, 5],
            sdk_collective_output_queues=[2, 4, 3, 5],
            dsr_dimension_ids=[1, 2],
            ownership="SDK x/y dimensions have separate resources; standalone profile, not implicit coexistence with SpMV",
        ),
        numeric_policy="finite f32, relaxed partitioned accumulation; stable norm uses global exponent scale before square-sum",
    )


def inputs(m, b):
    nodes = m["nodes"][:-2]
    check(set(b) == {n["host"] for n in nodes}, "reduction ports")
    values = []
    for n in nodes:
        v = b[n["host"]]
        check(
            len(v) == n["shape"][0]
            and all(
                type(x) in (int, float)
                and math.isfinite(x)
                and abs(x) <= m["input_bound"]
                and f32(x) == x
                for x in v
            ),
            "finite f32 reduction input",
        )
        values.append(v)
    return values


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "reduction epochs")
    result = []
    history = {n["id"]: [] for n in m["nodes"]}
    for b in batches:
        arrays = inputs(m, b)
        acc = 0.0
        if m["nodes"][-2]["op"] == "dot":
            for x, y in zip(*arrays):
                acc = f32(acc + f32(x * y))
        else:
            peak = max(map(abs, arrays[0]))
            alpha = 1.0 if peak == 0 else 2.0 ** max(-126, math.frexp(peak)[1] - 1)
            inv = f32(1.0 / alpha)
            for x in arrays[0]:
                z = f32(x * inv)
                acc = f32(acc + f32(z * z))
            acc = f32(f32(math.sqrt(acc)) * alpha)
        result.append({m["nodes"][-1]["host"]: [acc]})
    return result, history


def generate(s, dest):
    runtime = source_path("runtime")
    for name, template in TEMPLATES.items():
        (Path(dest) / name).write_text((runtime / template).read_text())
