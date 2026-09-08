"""Resident typed score-to-softmax composition with explicit resource lifetimes."""

import copy
from pathlib import Path
import numpy as np
from frontend import check
import mesh_score as score
import mesh_softmax as softmax


def children(m):
    q, k, t, mm, sm, out = copy.deepcopy(m["nodes"])
    sink = dict(
        id="__internal_score_sink",
        op="output",
        inputs=[mm["id"]],
        host="__resident_logits",
        shape=mm["shape"],
        dtype="f16",
    )
    # The synthetic sink/input only delimit verification fragments; never a host transfer.
    used = {n["id"] for n in m["nodes"]}
    while sink["id"] in used:
        sink["id"] += "_"
    a = copy.deepcopy(m)
    a["nodes"] = [q, k, t, mm, sink]
    inp = copy.deepcopy(mm)
    inp.update(op="input", inputs=[], host="__resident_logits")
    inp.pop("dataflow", None)
    b = copy.deepcopy(m)
    b["nodes"] = [inp, sm, out]
    return a, b


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    nodes = m["nodes"]
    by = {n["id"]: n for n in nodes}
    check(
        len(nodes) == len(by) == 6
        and sorted(n["op"] for n in nodes)
        == ["input", "input", "matmul", "output", "softmax", "transpose"],
        "score softmax six unique typed nodes",
    )
    check(
        all(v in by for n in nodes for v in n["inputs"]),
        "score softmax defined operands",
    )
    sm = next(n for n in nodes if n["op"] == "softmax")
    out = next(n for n in nodes if n["op"] == "output")
    check(
        len(sm["inputs"]) == 1 and out["inputs"] == [sm["id"]],
        "resident softmax result edge",
    )
    mm = by[sm["inputs"][0]]
    check(
        mm["op"] == "matmul" and len(mm["inputs"]) == 2,
        "resident softmax consumes contraction",
    )
    t = by[mm["inputs"][1]]
    check(
        t["op"] == "transpose" and len(t["inputs"]) == 1,
        "resident score transpose edge",
    )
    q = by[mm["inputs"][0]]
    k = by[t["inputs"][0]]
    m["nodes"] = [q, k, t, mm, sm, out]
    a, b = children(m)
    a = score.verify(a, epochs, bound)
    b = softmax.verify(b, epochs, q["shape"][1])
    m["nodes"] = a["nodes"][:4] + b["nodes"][1:]
    m.update(profile="mesh_score_softmax.v1", epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    a, b = children(m)
    a = score.verify(a, m["epochs"], m["input_bound"])
    b = softmax.verify(b, m["epochs"], m["nodes"][0]["shape"][1])
    s = score.plan(a, partitions)
    sm = softmax.plan(b, partitions)
    check(
        sm["rows"] == sm["cols"] == s["P"] and sm["Mt"] == sm["Nt"] == s["Mt"],
        "resident score/softmax tile compatibility",
    )
    check(
        sm["elementwise"] == "map", "resident softmax explicit SDK map implementation"
    )
    memory = copy.deepcopy(s["memory_per_pe"])
    memory["softmax_local_vectors"] = 4 * s["Mt"]
    memory["softmax_observations"] = (
        2 * (s["score_length"] + 5 * s["Mt"])
        if s["instrumentation"] == "sampled"
        else 4
    )
    memory["softmax_descriptors_control"] = 1024
    check(sum(memory.values()) <= 49152, "resident score softmax PE memory budget")
    out = copy.deepcopy(s)
    out.update(
        profile="mesh_score_softmax.v1",
        score_schedule=s,
        softmax_schedule=sm,
        scale=sm["scale"],
        memory_per_pe=memory,
        stages=s["stages"]
        + [
            "borrow completed partial buffer for exponents",
            "scale and local max; row maximum collective",
            "SDK map exp and local sum; row sum collective",
            "normalize and release command stream",
        ],
        ownership=dict(
            score="logical score tile becomes softmax in place",
            partial="all root reductions complete before partial storage becomes exponents",
            intermediate_host_transfer=False,
        ),
    )
    out["resources"]["active_input_queues"] = [3, 4, 5, 6]
    out["resources"]["active_output_queues"] = [3, 4, 5, 6]
    out["resources"][
        "reuse"
    ] = "score joins all vertical traffic before softmax borrows DSR1/2 and row queues3/4/6; no new colors/tasks/UTs"
    return out


def inputs(m, b):
    return score.inputs(m, b)


def evaluate(m, batches):
    a, b = children(m)
    a = score.verify(a, m["epochs"], m["input_bound"])
    b = softmax.verify(b, m["epochs"], m["nodes"][0]["shape"][1])
    values, _ = score.evaluate(a, batches)
    return softmax.evaluate(b, values)


def reference(s, q, k):
    partial, owners, roots, logits = score.reference(s["score_schedule"], q, k)
    history, exponents, result = softmax.reference(s["softmax_schedule"], logits)
    return partial, owners, roots, logits, history, exponents, result


def generate(s, dest):
    rt = Path(__file__).parent / "runtime"
    dest = Path(dest)
    for name in ("layout", "pe"):
        (dest / (name + ".csl")).write_bytes(
            (rt / ("score_softmax_" + name + ".csl")).read_bytes()
        )
    for name in ("inference_comm.csl", "inference_routes.csl", "softmax_local.csl"):
        (dest / name).write_bytes((rt / name).read_bytes())
    (dest / "WaferLLM-LICENSE.txt").write_bytes(
        (rt / "waferllm-LICENSE.txt").read_bytes()
    )
    (dest / "SOURCE-NOTICE.txt").write_text(
        "Source-derived resident score -> stable softmax, MeshInfra/WaferLLM fd1c2daae37cd68706c03fc8009887ecee9900f8, Apache-2.0. Unchanged communication module, explicit maximum initialization repair, shared synchronous CSL local-math library with SDK @map exp. Score partial storage reused only after completed root reductions. No intermediate host transfer, full attention or hardware claim.\n"
    )


def accuracy(s, q, k, actual):
    x = (q @ k.T) * s["scale"]
    v = np.exp(x - x.max(axis=1)[:, None])
    v /= v.sum(axis=1)[:, None]
    err = actual - v
    l2 = float(np.linalg.norm(err)) / float(np.linalg.norm(v))
    peak = float(np.max(np.abs(err))) / float(np.max(np.abs(v)))
    mass = float(np.max(np.abs(actual.sum(axis=1) - 1)))
    check(
        np.all(np.isfinite(actual))
        and np.all(actual >= 0)
        and l2 <= 0.015
        and peak <= 0.02
        and mass <= 0.01,
        "resident score softmax normwise accuracy",
    )
    return dict(
        contract="score-softmax-half-normwise-v1",
        fixed_accuracy_passed=True,
        relative_l2=l2,
        peak_scaled_error=peak,
        max_row_mass_error=mass,
    )
