"""Typed resident score/normalization/value composition over CSL phase contracts."""

import copy
from pathlib import Path
import numpy as np
from frontend import check
import mesh_score_softmax as normalization
import mesh_device_matmul as value


def children(m):
    q, k, v, t, mm, sm, pv, out = copy.deepcopy(m["nodes"])
    sink = copy.deepcopy(out)
    sink.update(
        id="__resident_probability_sink",
        inputs=[sm["id"]],
        shape=sm["shape"],
        host="__probability",
    )
    used = {n["id"] for n in m["nodes"]}
    while sink["id"] in used:
        sink["id"] += "_"
    hosts = {n.get("host") for n in m["nodes"]}
    while sink["host"] in hosts:
        sink["host"] += "_"
    first = copy.deepcopy(m)
    first["nodes"] = [q, k, t, mm, sm, sink]
    inp = copy.deepcopy(sm)
    inp.update(op="input", inputs=[], host=sink["host"])
    inp.pop("dataflow", None)
    second = copy.deepcopy(m)
    second["nodes"] = [inp, v, pv, out]
    return first, second


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    by = {n["id"]: n for n in m["nodes"]}
    check(
        len(by) == len(m["nodes"]) == 8
        and sorted(n["op"] for n in m["nodes"])
        == [
            "input",
            "input",
            "input",
            "matmul",
            "matmul",
            "output",
            "softmax",
            "transpose",
        ],
        "resident attention eight unique typed nodes",
    )
    check(
        all(i in by for n in m["nodes"] for i in n["inputs"]),
        "resident attention defined operands",
    )
    out = next(n for n in m["nodes"] if n["op"] == "output")
    check(len(out["inputs"]) == 1, "attention output edge")
    pv = by[out["inputs"][0]]
    check(
        pv["op"] == "matmul" and len(pv["inputs"]) == 2, "attention value contraction"
    )
    sm, v = [by[i] for i in pv["inputs"]]
    check(
        sm["op"] == "softmax" and len(sm["inputs"]) == 1 and v["op"] == "input",
        "attention resident probability edge",
    )
    mm = by[sm["inputs"][0]]
    check(
        mm["op"] == "matmul" and len(mm["inputs"]) == 2, "attention score contraction"
    )
    q, t = [by[i] for i in mm["inputs"]]
    check(
        t["op"] == "transpose" and len(t["inputs"]) == 1, "attention key transpose view"
    )
    k = by[t["inputs"][0]]
    check(
        q["op"] == k["op"] == "input" and len({q["id"], k["id"], v["id"]}) == 3,
        "attention distinct supplied inputs",
    )
    m["nodes"] = [q, k, v, t, mm, sm, pv, out]
    a, b = children(m)
    a = normalization.verify(a, epochs, bound)
    b = value.verify(b, epochs, 1)
    # Canonical fragments are only contracts, never SDK host transfers.
    m["nodes"] = [
        a["nodes"][0],
        a["nodes"][1],
        b["nodes"][1],
        a["nodes"][2],
        a["nodes"][3],
        a["nodes"][4],
        b["nodes"][2],
        b["nodes"][3],
    ]
    check(
        m["nodes"][0]["shape"] == m["nodes"][2]["shape"],
        "attention Q/K/V shared bounded shape",
    )
    m.update(profile="mesh_attention.v1", epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    a, b = children(m)
    a = normalization.verify(a, m["epochs"], m["input_bound"])
    b = value.verify(b, m["epochs"], 1)
    s = normalization.plan(a, partitions)
    v = value.plan(b, partitions)
    check(
        all(s[k] == v[k] for k in ("P", "M", "N", "Mt", "Nt", "instrumentation")),
        "attention shared geometry and observation mode",
    )
    memory = copy.deepcopy(s["memory_per_pe"])
    # K receive scratch is dead after score; share it with V receives. The S
    # exponent buffer is dead after normalization and becomes probability scratch.
    memory["value_input_work_output"] = 6 * s["length"]
    memory["value_observations"] = (
        2 * s["P"] * (s["score_length"] + 2 * s["length"]) + 4 * s["score_length"]
        if s["instrumentation"] == "sampled"
        else 10
    )
    memory["value_descriptors_control"] = 2048
    check(sum(memory.values()) <= 49152, "resident attention PE memory budget")
    out = copy.deepcopy(s)
    out.update(
        profile="mesh_attention.v1",
        normalization_schedule=s,
        value_schedule=v,
        memory_per_pe=memory,
    )
    out["stages"] = s["stages"][:-1] + [
        "normalize resident probability",
        "snapshot optional probability/exponent witnesses",
        "borrow completed exponent and K receive buffers for value contraction",
        "device V vertical and probability horizontal alignment",
        "overlap both-axis transfer with strided DSR value contraction",
        "join completion and return output",
    ]
    out["descriptor_entry_states"] = dict(
        score_right="contiguous stride1, reset every invocation after value phase",
        value_right="stride Mt, contraction increment1",
        compute="DSR1 reloaded for each local contraction; async communication uses distinct DSRs",
    )
    out["ownership"] = dict(
        intermediate_host_transfer=False,
        probability="score tile, then destructive left alignment",
        partial="score partial -> softmax exponent -> value left scratch",
        receive="completed K receive storage -> value receive scratch",
        public_inputs="immutable Q/K/V; private K/V copies",
    )
    out["resources"] = copy.deepcopy(v["resources"])
    out["resources"][
        "reuse"
    ] = "score and softmax drain communication before value reuses queues3..7, tasks19/20/25/26, UT0..3 and DSR1/3/4; phase dispatch preserves joins"
    return out


def inputs(m, b):
    check(set(b) == {n["host"] for n in m["nodes"][:3]}, "attention input ports")
    arrays = []
    for n in m["nodes"][:3]:
        a = np.asarray(b[n["host"]], float).reshape(n["shape"])
        check(
            np.all(np.isfinite(a))
            and np.all(np.abs(a) <= m["input_bound"])
            and np.array_equal(a, a.astype(np.float16).astype(float)),
            "attention finite exact-half bounded inputs",
        )
        arrays.append(a)
    return arrays


def evaluate(m, batches):
    a, b = children(m)
    a = normalization.verify(a, m["epochs"], m["input_bound"])
    b = value.verify(b, m["epochs"], 1)
    qs = [{n["host"]: batch[n["host"]] for n in a["nodes"][:2]} for batch in batches]
    probabilities, _ = normalization.evaluate(a, qs)
    intermediate_host = a["nodes"][-1]["host"]
    value_host = m["nodes"][2]["host"]
    vb = [
        {intermediate_host: p[intermediate_host], value_host: batch[value_host]}
        for p, batch in zip(probabilities, batches)
    ]
    return value.evaluate(b, vb)


def reference(s, q, k, v):
    first = normalization.reference(s["normalization_schedule"], q, k)
    second = value.reference(s["value_schedule"], first[-1], v)
    return (*first, *second)


def accuracy(s, q, k, v, actual):
    x = (q @ k.T) * s["scale"]
    p = np.exp(x - x.max(axis=1)[:, None])
    p /= p.sum(axis=1)[:, None]
    nominal = p @ v
    error = actual - nominal
    l2 = float(np.linalg.norm(error)) / max(float(np.linalg.norm(nominal)), 1e-30)
    peak = float(np.max(np.abs(error))) / max(float(np.max(np.abs(nominal))), 1e-30)
    check(
        np.all(np.isfinite(actual)) and l2 <= 0.02 and peak <= 0.025,
        "resident attention output normwise accuracy",
    )
    return dict(
        contract="attention-half-normwise-v1",
        fixed_accuracy_passed=True,
        relative_l2=l2,
        peak_scaled_error=peak,
    )


def generate(s, dest):
    rt = Path(__file__).parent / "runtime"
    dest = Path(dest)
    for n in ("pe", "layout"):
        (dest / (n + ".csl")).write_bytes(
            (rt / ("attention_" + n + ".csl")).read_bytes()
        )
    for n in ("inference_comm.csl", "inference_routes.csl", "softmax_local.csl"):
        (dest / n).write_bytes((rt / n).read_bytes())
    (dest / "WaferLLM-LICENSE.txt").write_bytes(
        (rt / "waferllm-LICENSE.txt").read_bytes()
    )
    (dest / "SOURCE-NOTICE.txt").write_text(
        "Source-derived resident unmasked supplied-Q/K/V single-head attention, MeshInfra/WaferLLM fd1c2daae37cd68706c03fc8009887ecee9900f8 Apache-2.0. Explicit maximum repair and device V alignment/strided DSD adaptation. Shared SDK local math and unchanged communication. No intermediate host transfer; not unmodified source, full model or hardware qualification.\n"
    )
