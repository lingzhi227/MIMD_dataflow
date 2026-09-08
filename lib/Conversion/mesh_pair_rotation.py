from source_tree import source_path, logical_name
"""Adjacent feature pair transform with explicit order and coefficient ownership."""

import copy
from pathlib import Path
import numpy as np
from frontend import check

POLICY = dict(partition="tiles", compute="dsd", fp="relaxed")


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    nodes = m["nodes"]
    check(
        len(nodes) == 5 and len({n["id"] for n in nodes}) == 5,
        "pair rotation five unique nodes",
    )
    check(
        sorted(n["op"] for n in nodes)
        == ["input", "input", "input", "output", "rotate_pairs"],
        "pair rotation typed subgraph",
    )
    by = {n["id"]: n for n in nodes}
    op = next(n for n in nodes if n["op"] == "rotate_pairs")
    out = next(n for n in nodes if n["op"] == "output")
    check(
        len(op["inputs"]) == 3
        and len(set(op["inputs"])) == 3
        and all(v in by for n in nodes for v in n["inputs"]),
        "pair rotation defined distinct edges",
    )
    x, c, s = [by[v] for v in op["inputs"]]
    check(
        all(n["op"] == "input" and not n["inputs"] for n in (x, c, s))
        and len({n["host"] for n in (x, c, s)}) == 3,
        "pair rotation input roles",
    )
    check(
        out["inputs"] == [op["id"]]
        and not m["states"]
        and not any("place" in n for n in nodes),
        "pair rotation region/output ownership",
    )
    check(
        all(n.get("dtype") == "f16" for n in (x, c, s, op)),
        "pair rotation half tensors",
    )
    M, N = x["shape"]
    check(
        all(type(v) is int and 1 <= v <= 4096 for v in (M, N)) and N % 2 == 0,
        "pair rotation even feature shape",
    )
    check(
        c["shape"] == s["shape"]
        and c["shape"] in ([1, N // 2], [M, N // 2])
        and op["shape"] == x["shape"],
        "pair rotation coefficient shape",
    )
    check(
        op["pair_order"] in ("even_odd", "odd_even"),
        "pair rotation explicit input order",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 8,
        "pair rotation bounded epochs and inputs",
    )
    check("dataflow" in op, "pair rotation explicit dataflow")
    out.update(shape=x["shape"][:], dtype="f16")
    m["nodes"] = [x, c, s, op, out]
    for n in nodes:
        n["interval"] = None
    m.update(profile="mesh_pair_rotation.v1", epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    x, c, s, op, out = m["nodes"]
    d = op["dataflow"]
    M, N = x["shape"]
    broadcast = c["shape"][0] == 1
    batched = d.get("layout") == "batch_major"
    if batched:
        check(
            set(d)
            == {
                "rows",
                "cols",
                "partition",
                "axis",
                "layout",
                "coefficients",
                "compute",
                "fp",
            }
            and d["partition"] == "features"
            and d["axis"] in ("x", "y")
            and d["coefficients"] == "feature_pairs"
            and broadcast
            and d["compute"] == "dsr"
            and d["fp"] == "relaxed",
            "batch pair rotation dataflow policy",
        )
    else:
        check(
            set(d) == set(POLICY) | {"rows", "cols", "coefficients"}
            and all(d[k] == v for k, v in POLICY.items())
            and d["coefficients"] == ("feature_pairs" if broadcast else "per_token"),
            "pair rotation dataflow policy",
        )
    rows, cols = d["rows"], d["cols"]
    check(
        partitions == 1 and all(type(v) is int and 1 <= v <= 16 for v in (rows, cols)),
        "pair rotation region bounds",
    )
    if batched:
        partitions_axis = cols if d["axis"] == "x" else rows
        check(
            N % partitions_axis == 0 and (N // partitions_axis) % 2 == 0,
            "batch pair unbroken adjacent features",
        )
        mt, nt = M, N // partitions_axis
    else:
        check(
            M % rows == 0 and N % cols == 0 and (N // cols) % 2 == 0,
            "pair rotation unbroken adjacent pairs in tiles",
        )
        mt, nt = M // rows, N // cols
    length = mt * nt
    check(
        not batched or 2 * length <= 32767, "batch pair signed descriptor offset bounds"
    )
    coefficient_length = (1 if broadcast else mt) * (nt // 2)
    mode = m.get("instrumentation", "sampled")
    check(
        length <= 32767 and mode in ("sampled", "counters"),
        "pair rotation DSD and observation bounds",
    )
    memory = dict(
        input_output_bytes=4 * length,
        coefficient_bytes=4 * coefficient_length,
        scratch_bytes=8 * (nt // 2 if batched else mt),
        observations_bytes=4 * length if mode == "sampled" else 2,
        control_code_stack_reserve=9216,
    )
    check(sum(memory.values()) <= 49152, "pair rotation PE memory budget")
    return dict(
        **(
            dict(
                layout="batch_major",
                axis=d["axis"],
                progress_extent=mt,
                local_kernel="batched_pair_rotation_local.csl",
            )
            if batched
            else {}
        ),
        profile=m["profile"],
        M=M,
        N=N,
        rows=rows,
        cols=cols,
        Mt=mt,
        Nt=nt,
        length=length,
        coefficient_length=coefficient_length,
        broadcast_coefficients=broadcast,
        pair_order=op["pair_order"],
        epochs=m["epochs"],
        instrumentation=mode,
        memory_per_pe=memory,
        stages=[
            dict(id=0, op="four_half_products", length=nt // 2 if batched else mt),
            dict(id=1, op="half_sub_add", pair_order=op["pair_order"]),
        ],
        resources=dict(
            colors=[],
            input_queues=[],
            output_queues=[],
            local_tasks=[],
            microthreads=[],
            explicit_dsr=list(range(1, 6)) if batched else [],
            ownership=(
                "Synchronous local borrow of dest/src0/src1 DSR banks 1..5 after prior SDK joins; four temporary lengths equal feature pairs; no async activity escapes apply. SDK owns I/O and launch."
                if batched
                else "Synchronous memory DSD arithmetic; all four temporary lengths equal token rows. SDK owns I/O and launch."
            ),
        ),
        storage=dict(
            order="batch-major" if batched else "column-major",
            inputs="immutable x/cosine/sine",
            result="separate output; paired features never split between PEs",
            coefficient_rows="replicated" if broadcast else "token-owned",
        ),
        nodes=[
            dict(id=f"p{x}_{y}", tile=[x, y], place=[4 + x, 1 + y])
            for y in range(rows)
            for x in range(cols)
        ],
    )


def inputs(m, b):
    nodes = m["nodes"][:3]
    check(set(b) == {n["host"] for n in nodes}, "pair rotation input ports")
    arrays = []
    for n in nodes:
        v = b[n["host"]]
        check(
            len(v) == np.prod(n["shape"]) and all(type(x) in (int, float) for x in v),
            "pair rotation input extent/type",
        )
        a = np.asarray(v, float).reshape(n["shape"])
        check(
            np.all(np.isfinite(a))
            and np.all(np.abs(a) <= m["input_bound"])
            and np.array_equal(a, a.astype(np.float16).astype(float)),
            "pair rotation finite exact-half input bounds",
        )
        arrays.append(a)
    return arrays


def reference(x, c, s, order, target=True):
    a, b = (x[:, ::2], x[:, 1::2]) if order == "even_odd" else (x[:, 1::2], x[:, ::2])
    q = lambda v: (
        np.asarray(v, np.float16).astype(float) if target else np.asarray(v, float)
    )
    products = [q(a * c), q(b * s), q(b * c), q(a * s)]
    out = np.empty_like(x)
    out[:, ::2] = q(products[0] - products[1])
    out[:, 1::2] = q(products[2] + products[3])
    return products, out


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "pair rotation epoch count")
    out = []
    for b in batches:
        value = (
            reference(*inputs(m, b), m["nodes"][3]["pair_order"], False)[1]
            .astype(np.float16)
            .astype(float)
        )
        out.append({m["nodes"][-1]["host"]: value.ravel().tolist()})
    return out, {}


def accuracy(x, c, s, order, actual):
    products, nominal = reference(x, c, s, order, False)
    magnitude = np.empty_like(x)
    magnitude[:, ::2] = np.abs(products[0]) + np.abs(products[1])
    magnitude[:, 1::2] = np.abs(products[2]) + np.abs(products[3])
    allowance = 0.0015 * magnitude + 2**-23
    error = np.abs(actual - nominal)
    check(
        np.all(np.isfinite(actual)) and np.all(error <= allowance),
        "pair rotation product-magnitude accuracy",
    )
    return dict(
        contract="pair-rotation-half-v1",
        fixed_accuracy_passed=True,
        max_abs_error=float(error.max()),
        max_error_over_allowance=float(np.max(error / allowance)),
        criterion=".0015*(abs(product0)+abs(product1))+2^-23 per output; cancellation-aware, not uniform relative error",
    )


def generate(s, dest):
    rt = source_path("runtime")
    dest = Path(dest)
    for name in ("layout", "pe"):
        (dest / (name + ".csl")).write_text(
            (
                rt
                / (
                    (
                        "batched_pair_rotation_"
                        if s.get("layout") == "batch_major" and name == "pe"
                        else "pair_rotation_"
                    )
                    + name
                    + ".csl"
                )
            ).read_text()
        )
    if s.get("layout") == "batch_major":
        (dest / "batched_pair_rotation_local.csl").write_bytes(
            (rt / "batched_pair_rotation_local.csl").read_bytes()
        )
    (dest / "WaferLLM-LICENSE.txt").write_bytes(
        (rt / "waferllm-LICENSE.txt").read_bytes()
    )
    if s.get("layout") == "batch_major":
        (dest / "SOURCE-NOTICE.txt").write_text(
            "Batch-major pair-vector schedule derived from MeshInfra/WaferLLM Decode xq_rope/xk_rope, commit fd1c2daae37cd68706c03fc8009887ecee9900f8, Apache-2.0. Explicit input pair order and feature-axis replicas, four packed-half DSR products before output stores; broadcast supplied coefficients. No position generation, cache update or full model claim.\n"
        )
        return
    (dest / "SOURCE-NOTICE.txt").write_text(
        "Pair-vector schedule inspired by MeshInfra/WaferLLM Prefill xq_rope, commit fd1c2daae37cd68706c03fc8009887ecee9900f8, Apache-2.0. Explicit input pair order; token-row-sized scratch; optional per-token coefficient vectors; immutable inputs; separate output and diagnostics. Standard rotation requires even_odd order and valid sine/cosine tables. No full RoPE model/inference claim.\n"
    )
