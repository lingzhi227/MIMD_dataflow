"""Typed graph verification, numeric contracts and backend-independent interpreter."""

import copy, math
from frontend import check, Error

LIMIT = 3.4028234e38
from float32 import f32, close


def interval(e, x):
    op = e[0]
    if op == "const":
        v = (e[1], e[1])
    elif op == "var":
        v = x
    elif op == "neg":
        a = interval(e[1], x)
        v = (-a[1], -a[0])
    else:
        a, b = interval(e[1], x), interval(e[2], x)
        if op == "+":
            v = (a[0] + b[0], a[1] + b[1])
        elif op == "-":
            v = (a[0] - b[1], a[1] - b[0])
        else:
            q = [i * j for i in a for j in b]
            v = (min(q), max(q))
    check(-LIMIT <= v[0] <= v[1] < LIMIT, "possible i32 intermediate overflow")
    return v


def verify(module, epochs, bound):
    from input_contracts import verify_declarations

    verify_declarations(module)
    if sum(n["op"] == "matmul" for n in module["nodes"]) == 3 and all(
        any(n["op"] == op for n in module["nodes"]) for op in ("rmsnorm", "silu", "add")
    ):
        from mesh_feed_forward import verify as feed_forward_verify

        return feed_forward_verify(module, epochs, bound)
    if all(
        any(n["op"] == op for n in module["nodes"])
        for op in ("matmul", "add", "rmsnorm")
    ):
        from mesh_projection_residual_rms import verify as composition_verify

        return composition_verify(module, epochs, bound)
    if sum(n["op"] == "matmul" for n in module["nodes"]) == 3 and any(
        n["op"] == "silu" for n in module["nodes"]
    ):
        from mesh_mlp import verify as mlp_verify

        return mlp_verify(module, epochs, bound)
    if (
        any(n["op"] == "softmax" for n in module["nodes"])
        and sum(n["op"] == "matmul" for n in module["nodes"]) == 2
    ):
        from mesh_attention import verify as attention_verify

        return attention_verify(module, epochs, bound)
    if any(
        n.get("dataflow", {}).get("initial_align") == "both_axes"
        for n in module["nodes"]
    ):
        from mesh_device_matmul import verify as device_verify

        return device_verify(module, epochs, bound)
    if any(n["op"] == "softmax" for n in module["nodes"]) and any(
        n["op"] == "matmul" for n in module["nodes"]
    ):
        from mesh_score_softmax import verify as resident_verify

        return resident_verify(module, epochs, bound)
    if any(n["op"] == "transpose" for n in module["nodes"]) and any(
        n.get("dtype") == "f16" for n in module["nodes"]
    ):
        from mesh_score import verify as score_verify

        return score_verify(module, epochs, bound)
    if any(n["op"] == "rotate_pairs" for n in module["nodes"]):
        from mesh_pair_rotation import verify as pair_verify

        return pair_verify(module, epochs, bound)
    if any(n["op"] == "silu" for n in module["nodes"]):
        from mesh_swiglu import verify as gating_verify

        return gating_verify(module, epochs, bound)
    if (
        any(n["op"] == "rmsnorm" for n in module["nodes"])
        and sum(n["op"] == "matmul" for n in module["nodes"]) > 1
    ):
        from mesh_normalized_fanout import verify as fanout_verify

        return fanout_verify(module, epochs, bound)
    if any(n["op"] == "rmsnorm" for n in module["nodes"]) and any(
        n["op"] == "matmul" for n in module["nodes"]
    ):
        from mesh_normalized_matmul import verify as resident_verify

        return resident_verify(module, epochs, bound)
    if any(n["op"] == "softmax" for n in module["nodes"]):
        from mesh_softmax import verify as softmax_verify

        return softmax_verify(module, epochs, bound)
    if any(n["op"] == "rmsnorm" for n in module["nodes"]):
        from mesh_rms import verify as rms_verify

        return rms_verify(module, epochs, bound)
    if any(n["op"] == "fft3d" for n in module["nodes"]):
        from mesh_fft import verify as fft_verify

        return fft_verify(module, epochs, bound)
    if any(n.get("dtype") == "f16" for n in module["nodes"]):
        if any(
            n.get("dataflow", {}).get("reduce") == "grouped_two_tree"
            for n in module["nodes"]
        ):
            from mesh_grouped_gemv import verify as half_verify
        else:
            check(
                any(
                    n.get("dataflow", {}).get("exchange") == "two_hop"
                    for n in module["nodes"]
                ),
                "f16 requires a qualified typed spatial lowering",
            )
            from mesh_twohop import verify as half_verify
        return half_verify(module, epochs, bound)
    if any(n["op"] == "power_csc" for n in module["nodes"]):
        from power_ir import verify as power_verify

        return power_verify(module, epochs, bound)
    if any(n["op"] in ("cg_csc", "pcg_csc", "bicgstab_csc") for n in module["nodes"]):
        from solver_ir import verify as solver_verify

        return solver_verify(module, epochs, bound)
    if any("dataflow" in n for n in module["nodes"]):
        if any(n["op"] in ("dot", "nrm2") for n in module["nodes"]):
            from mesh_reduction import verify as mesh_verify
        elif any(n["op"] == "spmv_csc" for n in module["nodes"]):
            from mesh_spmv import verify as mesh_verify
        elif any(n["op"] == "qr_r" for n in module["nodes"]):
            from mesh_qr import verify as mesh_verify
        elif any(n["op"] == "lu_no_pivot" for n in module["nodes"]):
            from mesh_lu import verify as mesh_verify
        elif any(n["op"] == "cholesky" for n in module["nodes"]):
            from mesh_cholesky import verify as mesh_verify
        elif any(
            n.get("dataflow", {}).get("exchange") == "cyclic" for n in module["nodes"]
        ):
            from mesh_cannon import verify as mesh_verify
        elif any(
            n.get("dataflow", {}).get("broadcast") == "rows_columns"
            for n in module["nodes"]
        ):
            from mesh_gemm import verify as mesh_verify
        else:
            from mesh_gemv import verify as mesh_verify

        return mesh_verify(module, epochs, bound)
    if any(n["op"] == "grid_iterate" for n in module["nodes"]):
        from grid_ir import verify as grid_verify

        return grid_verify(module, epochs, bound)
    m = copy.deepcopy(module)
    check(type(epochs) is int and 1 <= epochs <= 16, "epochs 1..16")
    check(type(bound) is int and 0 <= bound <= 32767, "bound 0..32767")
    known = {}
    hosts = {"input": set(), "output": set()}
    writers = set()
    for n in m["nodes"]:
        op = n["op"]
        check(n["id"] not in known, "duplicate id")
        check(all(i in known for i in n["inputs"]), "forward dependency")
        ins = [known[i] for i in n["inputs"]]
        if op in hosts:
            check(n["host"] not in hosts[op], "duplicate host port")
            hosts[op].add(n["host"])
        if op == "input":
            pass
        elif op == "kernel":
            check(len(ins) == len(n["body"]["params"]), "kernel arguments")
            for p, v in zip(n["body"]["params"], ins):
                check(n["body"]["arrays"][p] == v["shape"], "kernel input shape")
        elif op in ("map", "accumulate", "output", "transpose", "row_sum"):
            r, c = ins[0]["shape"]
            wanted = {"transpose": [c, r], "row_sum": [r, 1]}.get(op, [r, c])
            if op == "output":
                n["shape"] = wanted
            check(n["shape"] == wanted, "shape mismatch")
            if op == "accumulate":
                key = n["state"]
                check(
                    key in m["states"]
                    and m["states"][key] == wanted
                    and key not in writers,
                    "state contract",
                )
                writers.add(key)
        elif op == "add":
            check(ins[0]["shape"] == ins[1]["shape"] == n["shape"], "add shape")
        elif op == "matmul":
            r, k = ins[0]["shape"]
            k2, c = ins[1]["shape"]
            check(k == k2 and n["shape"] == [r, c], "matmul shape")
        else:
            raise Error("unknown operation " + op)
        check(
            len(n["shape"]) == 2
            and all(type(x) is int and x > 0 for x in n["shape"])
            and math.prod(n["shape"]) <= 256,
            "tensor extent 1..256",
        )
        n["interval"] = None
        n["numeric_policy"] = (
            "f32; input bound; guarded accesses; finite outputs; declared tolerance"
        )
        known[n["id"]] = n
    check(writers == set(m["states"]), "unused state")
    check(hosts["input"] and hosts["output"], "host ports required")
    live = set()

    def visit(i):
        if i not in live:
            live.add(i)
            for x in known[i]["inputs"]:
                visit(x)

    for n in m["nodes"]:
        if n["op"] == "output":
            visit(n["id"])
    check(live == set(known), "unobserved computation")
    m.update(epochs=epochs, input_bound=bound)
    return m


def eval_expr(e, x):
    if e[0] == "const":
        return e[1]
    if e[0] == "var":
        return x
    if e[0] == "neg":
        return -eval_expr(e[1], x)
    a, b = eval_expr(e[1], x), eval_expr(e[2], x)
    return f32({"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b}[e[0]]())


def evaluate(m, batches):
    if m.get("profile") == "mesh_feed_forward.v1":
        from mesh_feed_forward import evaluate as feed_forward_evaluate

        return feed_forward_evaluate(m, batches)
    if m.get("profile") == "mesh_projection_residual_rms.v1":
        from mesh_projection_residual_rms import evaluate as composition_evaluate

        return composition_evaluate(m, batches)
    if m.get("profile") == "mesh_normalized_fanout.v1":
        from mesh_normalized_fanout import evaluate as fanout_evaluate

        return fanout_evaluate(m, batches)
    if m.get("profile") == "mesh_pair_rotation.v1":
        from mesh_pair_rotation import evaluate as pair_evaluate

        return pair_evaluate(m, batches)
    if m.get("profile") == "mesh_mlp.v1":
        from mesh_mlp import evaluate as mlp_evaluate

        return mlp_evaluate(m, batches)
    if m.get("profile") == "mesh_attention.v1":
        from mesh_attention import evaluate as attention_evaluate

        return attention_evaluate(m, batches)
    if m.get("profile") == "mesh_score_softmax.v1":
        from mesh_score_softmax import evaluate as resident_evaluate

        return resident_evaluate(m, batches)
    if m.get("profile") == "mesh_device_matmul.v1":
        from mesh_device_matmul import evaluate as device_evaluate

        return device_evaluate(m, batches)
    if m.get("profile") == "mesh_score.v1":
        from mesh_score import evaluate as score_evaluate

        return score_evaluate(m, batches)
    if m.get("profile") == "mesh_swiglu.v1":
        from mesh_swiglu import evaluate as gating_evaluate

        return gating_evaluate(m, batches)
    if m.get("profile") == "mesh_normalized_matmul.v1":
        from mesh_normalized_matmul import evaluate as resident_evaluate

        return resident_evaluate(m, batches)
    if m.get("profile") == "mesh_softmax.v1":
        from mesh_softmax import evaluate as softmax_evaluate

        return softmax_evaluate(m, batches)
    if m.get("profile") == "mesh_rms.v1":
        from mesh_rms import evaluate as rms_evaluate

        return rms_evaluate(m, batches)
    if m.get("profile") == "mesh_fft.v1":
        from mesh_fft import evaluate as fft_evaluate

        return fft_evaluate(m, batches)
    if m.get("profile") in ("mesh_twohop.v1", "mesh_grouped_gemv.v1"):
        from half_matrix import evaluate as half_evaluate

        return half_evaluate(m, batches)
    if m.get("profile") == "mesh_power.v1":
        from power_ir import evaluate as power_evaluate

        return power_evaluate(m, batches)
    if m.get("profile") == "mesh_cg.v1":
        from solver_reference import evaluate as cg_evaluate

        return cg_evaluate(m, batches)
    if m.get("profile") == "mesh_reduction.v1":
        from mesh_reduction import evaluate as reduce_evaluate

        return reduce_evaluate(m, batches)
    if m.get("profile") == "mesh_spmv.v1":
        from mesh_spmv import evaluate as sparse_evaluate

        return sparse_evaluate(m, batches)
    if m.get("profile") == "grid.v1":
        from grid_ir import evaluate as grid_evaluate

        return grid_evaluate(m, batches)
    check(len(batches) == m["epochs"], "epoch count")
    state = {k: [0] * math.prod(s) for k, s in m["states"].items()}
    result = []
    history = {n["id"]: [] for n in m["nodes"]}
    by = {n["id"]: n for n in m["nodes"]}
    for batch in batches:
        check(
            set(batch) == {n["host"] for n in m["nodes"] if n["op"] == "input"},
            "host input set",
        )
        vals = {}
        outputs = {}
        for n in m["nodes"]:
            op = n["op"]
            a = [vals[i] for i in n["inputs"]]
            r, c = n["shape"]
            if op == "input":
                v = batch[n["host"]]
                check(
                    len(v) == r * c
                    and all(
                        type(x) in (int, float) and abs(x) <= m["input_bound"]
                        for x in v
                    ),
                    "host input contract",
                )
            elif op == "fork":
                v = a[0]
            elif op == "slice":
                sr, sc = by[n["inputs"][0]]["shape"]
                v = [
                    a[0][
                        (i + (n["start"] if n["axis"] == 0 else 0)) * sc
                        + j
                        + (n["start"] if n["axis"] == 1 else 0)
                    ]
                    for i in range(r)
                    for j in range(c)
                ]
            elif op == "kernel":
                from local_kernel import evaluate_kernel

                v = evaluate_kernel(n["body"], a)
            elif op == "map":
                v = [f32(eval_expr(n["expr"], x)) for x in a[0]]
            elif op == "add":
                v = [f32(x + y) for x, y in zip(*a)]
            elif op == "transpose":
                v = [a[0][j * r + i] for i in range(r) for j in range(c)]
            elif op == "row_sum":
                k = by[n["inputs"][0]]["shape"][1]
                v = [fsum(a[0][i * k : (i + 1) * k]) for i in range(r)]
            elif op == "qr_r":
                from mesh_qr import factor

                v = factor(a[0], r, c)
            elif op == "lu_no_pivot":
                from mesh_lu import factor

                v = factor(a[0], r)
            elif op == "cholesky":
                from mesh_cholesky import factor_f32

                v = factor_f32(a[0], r)
            elif op == "matmul":
                k = by[n["inputs"][0]]["shape"][1]
                v = [
                    fsum(f32(a[0][i * k + t] * a[1][t * c + j]) for t in range(k))
                    for i in range(r)
                    for j in range(c)
                ]
            elif op == "accumulate":
                v = [f32(x + y) for x, y in zip(state[n["state"]], a[0])]
                state[n["state"]] = v
            elif op == "output":
                v = a[0]
                outputs[n["host"]] = v
            vals[n["id"]] = [f32(x) for x in v]
            history[n["id"]].extend(v)
        result.append(outputs)
    return result, history


def fsum(values):
    result = 0.0
    for v in values:
        result = f32(result + v)
    return result
