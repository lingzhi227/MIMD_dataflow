"""Resident rectangular projection/gating composition using ordinary typed operators."""

import copy, math
from pathlib import Path
import numpy as np
from frontend import check
from projection_contract import stage
from input_contracts import effective_bound, half_dot_bound, half_product_bound
from mesh_swiglu import POLICY, standard, reference as gating_reference
from binary16 import matmul


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    ns = m["nodes"]
    by = {n["id"]: n for n in ns}
    check(
        len(by) == len(ns) == 10
        and sorted(n["op"] for n in ns)
        == sorted(["input"] * 4 + ["matmul"] * 3 + ["silu", "multiply", "output"]),
        "MLP ten unique typed nodes",
    )
    check(all(i in by for n in ns for i in n["inputs"]), "MLP defined operands")
    out = next(n for n in ns if n["op"] == "output")

    def operands(n, op, count):
        check(n["op"] == op and len(n["inputs"]) == count, "MLP " + op + " edges")
        return [by[i] for i in n["inputs"]]

    (down,) = operands(out, "output", 1)
    hidden, wd = operands(down, "matmul", 2)
    pair = operands(hidden, "multiply", 2)
    check(sum(n["op"] == "silu" for n in pair) == 1, "MLP gating edge")
    act = next(n for n in pair if n["op"] == "silu")
    up = next(n for n in pair if n is not act)
    (gate,) = operands(act, "silu", 1)
    x, wu = operands(up, "matmul", 2)
    xx, wg = operands(gate, "matmul", 2)
    check(
        x["id"] == xx["id"] and len({n["id"] for n in [x, wu, wg, wd]}) == 4,
        "MLP shared activation and distinct weights",
    )
    check(
        all(n["op"] == "input" and not n["inputs"] for n in [x, wu, wg, wd])
        and len({n["host"] for n in [x, wu, wg, wd]}) == 4,
        "MLP input ports",
    )
    check(not m["states"] and not any("place" in n for n in ns), "MLP owns mesh region")
    check(all(n.get("dtype") == "f16" for n in ns if n is not out), "MLP half tensors")
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 1,
        "MLP epoch/global input bound",
    )
    check(
        wu["shape"] == wg["shape"] and wd["shape"] == wu["shape"][::-1],
        "MLP expansion/contraction weights",
    )
    check(
        up["shape"]
        == gate["shape"]
        == act["shape"]
        == hidden["shape"]
        == [x["shape"][0], wu["shape"][1]]
        and down["shape"] == x["shape"],
        "MLP intermediate shapes",
    )
    out.update(shape=down["shape"][:], dtype="f16")
    m["nodes"] = [x, wu, wg, wd, up, gate, act, hidden, down, out]
    for n in m["nodes"]:
        n["interval"] = None
    m.update(profile="mesh_mlp.v1", epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    x, wu, wg, wd, up, gate, act, hidden, down, out = m["nodes"]
    check(partitions == 1, "MLP single mesh region")
    stages = [
        stage(x["shape"], wu["shape"], up["dataflow"]),
        stage(x["shape"], wg["shape"], gate["dataflow"]),
        stage(hidden["shape"], wd["shape"], down["dataflow"]),
    ]
    p = stages[0]["P"]
    M, N = x["shape"]
    F = wu["shape"][1]
    check(
        F >= N and M <= 256 and N <= 256 and all(v["P"] == p for v in stages),
        "MLP common bounded expansion mesh",
    )
    check(
        act.get("dataflow") == hidden.get("dataflow") == dict(POLICY, rows=p, cols=p),
        "MLP phase-exclusive CSL map/DSR gating",
    )
    bounds = [effective_bound(n, m["input_bound"]) for n in (x, wu, wg, wd)]
    ub = half_dot_bound(bounds[0], bounds[1], N)
    gb = half_dot_bound(bounds[0], bounds[2], N)
    check(
        max(ub, gb) <= 8,
        "MLP projected gate/up must satisfy validated SDK SiLU domain; refine input bounds",
    )
    hb = half_product_bound(ub, gb)
    ob = half_dot_bound(hb, bounds[3], F)
    mt, nt, ft = M // p, N // p, F // p
    L, H, W = mt * nt, mt * ft, nt * ft
    mode = m.get("instrumentation", "sampled")
    check(mode in ("sampled", "counters"), "MLP observation mode")
    memory = dict(
        resident_arrays=2 * (3 * L + 5 * W + 3 * H),
        observations=(
            2 * (p * (2 * H + L) + 3 * H + 2 * L + H + 3 * W)
            if mode == "sampled"
            else 16
        ),
        descriptors_control_reserve=2048,
        sdk_code_tasks_stack_reserve=16384,
    )
    check(sum(memory.values()) <= 49152, "MLP PE memory budget")
    return dict(
        profile="mesh_mlp.v1",
        M=M,
        N=N,
        F=F,
        P=p,
        rows=p,
        cols=p,
        Mt=mt,
        Nt=nt,
        Ft=ft,
        length=L,
        hidden_length=H,
        weight_length=W,
        epochs=m["epochs"],
        instrumentation=mode,
        projection_stages=stages,
        memory_per_pe=memory,
        numerical_bounds=dict(
            inputs=bounds,
            up=ub,
            gate=gb,
            hidden=hb,
            output=ob,
            reason="monotone half-FMA bounds; validated SDK half SiLU magnitude <= gate magnitude",
        ),
        stages=[
            "copy immutable X/U and forward-align X",
            "up contraction with joined double buffers",
            "copy G and carry completed left ownership",
            "gate contraction with joined double buffers",
            "resident SDK map SiLU and DSR in-place multiply",
            "copy D and forward-align hidden",
            "down contraction into dead X work storage",
            "drain queues and return",
        ],
        descriptor_entry_states=dict(
            right="contiguous stride1 reset on every projection entry; increment Nt",
            compute="DSR1 loaded for each contraction; gating only after both-axis join",
        ),
        ownership=dict(
            intermediate_host_transfer=False,
            inputs="immutable X/U/G/D",
            up="becomes hidden after last up read",
            x_work="becomes output after gate completes",
            weights="one private work/receive pair reused after each joined projection",
            gate="becomes activated gate in place",
        ),
        resources=dict(
            colors=list(range(1, 12)),
            input_queues=list(range(3, 8)),
            output_queues=list(range(3, 8)),
            local_tasks=[19, 20, 25, 26],
            microthreads=list(range(4)),
            compute_dsr=1,
            communication_dsrs=[3, 4],
            completion="two-axis completion joins before next step; phase-exclusive reuse",
        ),
        nodes=[
            dict(id=f"p{x}_{y}", tile=[x, y], place=[4 + x, 1 + y])
            for y in range(p)
            for x in range(p)
        ],
    )


def inputs(m, b):
    check(set(b) == {n["host"] for n in m["nodes"][:4]}, "MLP input ports")
    arrays = []
    for n in m["nodes"][:4]:
        a = np.asarray(b[n["host"]], float).reshape(n["shape"])
        check(
            np.all(np.isfinite(a))
            and np.all(np.abs(a) <= effective_bound(n, m["input_bound"]))
            and np.array_equal(a, a.astype(np.float16).astype(float)),
            "MLP finite exact-half contracted inputs",
        )
        arrays.append(a)
    return arrays


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "MLP epochs")
    results = []
    for b in batches:
        x, u, g, d = inputs(m, b)
        up = matmul(x, u)
        gate = matmul(x, g)
        act = standard(up, gate)[0].astype(np.float16).astype(float)
        hidden = (up * act).astype(np.float16).astype(float)
        results.append({m["nodes"][-1]["host"]: matmul(hidden, d).ravel().tolist()})
    return results, {}


def reference(s, x, u, g, d):
    from projection_reference import project

    up, uh, ul, ur = project(s["projection_stages"][0], x, u)
    gate, gh, gl, gr = project(s["projection_stages"][1], x, g)
    act, hidden = gating_reference(up, gate)
    out, dh, dl, dr = project(s["projection_stages"][2], hidden, d)
    from mesh_common import pack_tiles

    pack = lambda a: pack_tiles(a, s["P"], s["P"], "F")
    return out, dict(
        up_history=uh,
        gate_history=gh,
        down_history=dh,
        gate_snapshot=pack(gate),
        hidden_snapshot=pack(hidden),
        activated_gate=pack(act),
        left_first=np.concatenate([ul, gl, dl], axis=-1),
        right_first=np.concatenate([ur, gr, dr], axis=-1),
    )


def accuracy(s, x, u, g, d, actual):
    up = x @ u
    gate = x @ g
    nominal = standard(up, gate)[1] @ d
    error = actual - nominal
    l2 = float(np.linalg.norm(error)) / max(float(np.linalg.norm(nominal)), 1e-30)
    peak = float(np.max(np.abs(error))) / max(float(np.max(np.abs(nominal))), 1e-30)
    check(
        np.all(np.isfinite(actual)) and l2 <= 0.02 and peak <= 0.03,
        "MLP original-input normwise accuracy",
    )
    return dict(
        contract="rectangular-mlp-half-normwise-v1",
        fixed_accuracy_passed=True,
        relative_l2=l2,
        peak_scaled_error=peak,
    )


def generate(s, dest):
    rt = Path(__file__).parent / "runtime"
    dest = Path(dest)
    for n in ("pe", "layout"):
        (dest / (n + ".csl")).write_bytes((rt / ("mlp_" + n + ".csl")).read_bytes())
    for n in ("inference_comm.csl", "inference_routes.csl", "gated_local.csl"):
        (dest / n).write_bytes((rt / n).read_bytes())
    (dest / "WaferLLM-LICENSE.txt").write_bytes(
        (rt / "waferllm-LICENSE.txt").read_bytes()
    )
    (dest / "SOURCE-NOTICE.txt").write_text(
        "Source-derived rectangular gated MLP; MeshInfra/WaferLLM fd1c2daae37cd68706c03fc8009887ecee9900f8 Apache-2.0. Explicit gate-branch left ownership repair, immutable public inputs and last-use storage reuse. Unchanged communication and source SDK map SiLU/DSR arithmetic. Supplied X/U/G/D, no RMS/residual/full-model or hardware qualification.\n"
    )
