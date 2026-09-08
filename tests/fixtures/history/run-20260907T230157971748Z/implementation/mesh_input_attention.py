"""Development typed plan for the executed source-shaped31-node numerical chain.

Shared CSL generation exists; regular dispatch remains rejected while the full
eight-case numerical contract fails. Diagnostic execution is never qualification.
"""

import copy
import numpy as np
from frontend import check
from input_contracts import effective_bound
from rms_projection_bounds import projection as projection_bound
from pair_rotation_bounds import bound as pair_bound
from projection_contract import stage as projection_stage
from mesh_attention_tail import verify as tail_verify, plan as tail_plan
from mesh_rms import verify as rms_verify
from mesh_pair_rotation import verify as pair_verify


def _host(nodes, stem):
    used = {n.get("host") for n in nodes}
    while stem in used:
        stem += "_"
    return stem


def _id(nodes, stem):
    used = {n["id"] for n in nodes}
    while stem in used:
        stem += "_"
    return stem


def _prefix_bounds(x, gamma, weights, cosine, sine, norm, global_bound):
    p = norm["dataflow"]["cols"]
    n = x["shape"][1]
    result = {
        name: projection_bound(
            effective_bound(x, global_bound),
            effective_bound(gamma, global_bound),
            effective_bound(weight, global_bound),
            n // p,
            p,
            norm["epsilon"],
        )
        for name, weight in zip(("q", "k", "v"), weights)
    }
    for name in ("q", "k"):
        result[name]["pair"] = pair_bound(
            result[name]["projection_absolute"],
            effective_bound(cosine, global_bound),
            effective_bound(sine, global_bound),
        )
    return result


def bounds(m):
    ns = m["nodes"]
    return _prefix_bounds(ns[0], ns[1], ns[2:5], ns[5], ns[6], ns[11], m["input_bound"])


def tail(m):
    ns = copy.deepcopy(m["nodes"])
    r = bounds(m)
    values = []
    for name, idx in (("q", 15), ("k", 16), ("v", 14)):
        node = ns[idx]
        limit = (
            r[name]["pair"]["output_absolute"]
            if name != "v"
            else r[name]["projection_absolute"]
        )
        node.update(
            op="input", inputs=[], host=_host(ns, "__computed_" + name), abs_bound=limit
        )
        node.pop("dataflow", None)
        node.pop("pair_order", None)
        values.append(node)
    return dict(
        m,
        nodes=values + [ns[i] for i in (7, 0, 1, 8, 9, 10)] + ns[17:],
        profile="mesh_attention_tail.v1",
    )


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    ns = m["nodes"]
    by = {n["id"]: n for n in ns}
    check(len(ns) == len(by) == 31, "31 unique input-attention nodes")
    check(
        all(v in by for n in ns for v in n["inputs"]), "defined input-attention edges"
    )
    check(
        sorted(n["op"] for n in ns)
        == sorted(
            ["input"] * 11
            + ["matmul"] * 9
            + ["rmsnorm"] * 2
            + ["rotate_pairs"] * 2
            + ["add"] * 2
            + ["transpose", "softmax", "silu", "multiply", "output"]
        ),
        "input-attention operation inventory",
    )

    def operands(node, op, count):
        check(
            node["op"] == op and len(node["inputs"]) == count, op + " prefix operands"
        )
        return [by[i] for i in node["inputs"]]

    soft = next(n for n in ns if n["op"] == "softmax")
    (score,) = operands(soft, "softmax", 1)
    q, kt = operands(score, "matmul", 2)
    (k,) = operands(kt, "transpose", 1)
    qraw, c, s = operands(q, "rotate_pairs", 3)
    kraw, c2, s2 = operands(k, "rotate_pairs", 3)
    check(
        c["id"] == c2["id"] and s["id"] == s2["id"], "shared source pair coefficients"
    )
    norm, qw = operands(qraw, "matmul", 2)
    norm2, kw = operands(kraw, "matmul", 2)
    value = next(n for n in ns if n["op"] == "matmul" and n["inputs"][0] == soft["id"])
    _, v = operands(value, "matmul", 2)
    norm3, vw = operands(v, "matmul", 2)
    check(norm["id"] == norm2["id"] == norm3["id"], "shared normalized QKV activation")
    x, gamma = operands(norm, "rmsnorm", 2)
    prefix_inputs = [x, gamma, qw, kw, vw, c, s]
    check(
        all(n["op"] == "input" and not n["inputs"] for n in prefix_inputs),
        "source prefix public inputs",
    )
    check(
        len({n["id"] for n in prefix_inputs})
        == len({n["host"] for n in prefix_inputs})
        == 7,
        "distinct prefix input roles",
    )
    dims = x["shape"]
    n = dims[1]
    check(
        c["shape"] == s["shape"] == [1, n // 2] and n % 2 == 0,
        "source feature-broadcast coefficient shape",
    )
    for op in (q, k):
        check(
            op["pair_order"] == "odd_even",
            "source-shaped prefix explicit odd_even order",
        )
    for weight, op in ((qw, qraw), (kw, kraw), (vw, v)):
        check(
            weight["shape"] == [n, n] and op["shape"] == dims and op["dtype"] == "f16",
            "square QKV projection extent/type",
        )
        check(
            "accumulation" not in op["dataflow"],
            "source prefix ordinary half recurrence",
        )
        projection_stage(dims, weight["shape"], op["dataflow"])
    # Infer remaining tail roles through its established structural verifier.
    excluded = {v["id"] for v in (norm, qraw, kraw, qw, kw, vw, c, s)}
    limits = _prefix_bounds(x, gamma, (qw, kw, vw), c, s, norm, bound)
    derived = {
        q["id"]: limits["q"]["pair"]["output_absolute"],
        k["id"]: limits["k"]["pair"]["output_absolute"],
        v["id"]: limits["v"]["projection_absolute"],
    }
    child_nodes = [copy.deepcopy(n) for n in ns if n["id"] not in excluded]
    for node in child_nodes:
        if node["id"] in (q["id"], k["id"], v["id"]):
            node.update(
                op="input",
                inputs=[],
                host=_host(ns, "__probe_" + node["id"]),
                abs_bound=derived[node["id"]],
            )
            node.pop("dataflow", None)
            node.pop("pair_order", None)
    # Structural child verification uses the computed prefix range, never a guessed supplied-input envelope.
    child = tail_verify(dict(m, nodes=child_nodes), epochs, bound)
    check(
        child["nodes"][4]["id"] == x["id"] and child["nodes"][5]["id"] == gamma["id"],
        "source original X residual and shared gamma",
    )
    check(norm["epsilon"] == child["nodes"][15]["epsilon"], "source shared RMS epsilon")
    canonical = [
        x,
        gamma,
        qw,
        kw,
        vw,
        c,
        s,
        child["nodes"][3],
        *child["nodes"][6:9],
        norm,
        qraw,
        kraw,
        v,
        q,
        k,
        *child["nodes"][9:],
    ]
    check(
        len(canonical) == 31 and {n["id"] for n in canonical} == set(by),
        "all input-attention nodes consumed",
    )
    m.update(
        nodes=canonical,
        profile="mesh_input_attention.v1",
        epochs=epochs,
        input_bound=bound,
    )
    sink = dict(
        copy.deepcopy(ns[-1]),
        id=_id(ns, "__prefix_norm_sink"),
        op="output",
        inputs=[norm["id"]],
        host=_host(ns, "__prefix_normalized"),
        shape=dims,
        dtype="f16",
    )
    rms_verify(dict(m, nodes=copy.deepcopy([x, gamma, norm, sink])), epochs, bound)
    for op, raw in ((q, qraw), (k, kraw)):
        source = copy.deepcopy(raw)
        source.update(op="input", inputs=[], host=_host(ns, "__prefix_pair_input"))
        source.pop("dataflow", None)
        sink2 = dict(sink, id=_id(ns, "__prefix_pair_sink"), inputs=[op["id"]])
        pair_verify(
            dict(m, nodes=copy.deepcopy([source, c, s, op, sink2])), epochs, bound
        )
    tail_verify(tail(m), epochs, bound)
    for node in canonical:
        node["interval"] = None
    plan(m)
    return m


def plan(m, partitions=1):
    s = tail_plan(tail(m), partitions)
    ns = m["nodes"]
    p = s["P"]
    l = s["length"]
    ow = s["output_weight_length"]
    mt = s["Mt"]
    nt = s["Nt"]
    stages = [
        projection_stage(ns[11]["shape"], ns[i]["shape"], ns[j]["dataflow"])
        for i, j in ((2, 12), (3, 13), (4, 14))
    ]
    check(all(v["P"] == p for v in stages), "one prefix/tail mesh")
    check(
        all(
            op["dataflow"]["rows"] == p and op["dataflow"]["cols"] == p
            for op in (ns[11], ns[15], ns[16])
        ),
        "one RMS/pair/tail mesh",
    )
    check(l >= 4 * mt, "four pair scratch rows fit dead normalized workspace")
    sampled = s["instrumentation"] == "sampled"
    diagnostic = 3 * l + (3 * p * l + 7 * l + 3 * ow if sampled else 4)
    s["memory_per_pe"]["input_prefix_weights_coefficients_observers"] = (
        2 * (3 * ow + nt + diagnostic) + 32
    )
    s["memory_per_pe"]["input_prefix_code_descriptor_reserve"] = 2048
    check(sum(s["memory_per_pe"].values()) <= 49152, "input-attention PE memory budget")
    s.update(
        profile="mesh_input_attention.v1",
        input_prefix_stages=stages,
        input_prefix_bounds=bounds(m),
        input_bindings={
            node["host"]: port
            for node, port in zip(
                ns[:11],
                (
                    "residual",
                    "gamma",
                    "q_weight",
                    "k_weight",
                    "v_weight",
                    "cosine",
                    "sine",
                    "output_weight",
                    "up_weight",
                    "gate_weight",
                    "down_weight",
                ),
            )
        },
        input_composition=dict(
            expression="Xn=RMS(X,gamma); Q,K,V=Xn*Wq/k/v; rotate Q,K; Z=attention*O+X; Y=Z+MLP(RMS(Z,gamma))",
            public_residual="original X",
            shared_gamma=True,
            intermediate_host_transfer=False,
            projection_phases=[6, 7, 8],
            pair_order="odd_even",
            completion="join V and synchronous Q/K pair transforms before resident score",
            normalized_owner="one initial alignment, preserve completed live left pointer at K/V entries",
        ),
    )
    from input_attention_lifetimes import plan as lifetime_plan

    s["storage_lifetimes"] = lifetime_plan(s)
    return s


def inputs(m, batch):
    check(
        set(batch) == {v["host"] for v in m["nodes"][:11]},
        "eleven input-attention ports",
    )
    result = []
    for v in m["nodes"][:11]:
        a = np.asarray(batch[v["host"]], float)
        check(a.size == np.prod(v["shape"]), "input-attention input extent")
        a = a.reshape(v["shape"])
        check(
            np.all(np.isfinite(a))
            and np.all(np.abs(a) <= effective_bound(v, m["input_bound"]))
            and np.array_equal(a, a.astype(np.float16).astype(float)),
            "finite bounded exact-half input-attention inputs",
        )
        result.append(a)
    return result


def generate(s, dest):
    from pathlib import Path
    import shutil
    from mesh_attention_tail import generate as generate_tail
    from input_attention_csl import hooks, EXPORTS

    generate_tail(s, dest, composition_hooks=hooks)
    dest = Path(dest)
    shutil.copyfile(
        Path(__file__).parent / "runtime/pair_rotation_local.csl",
        dest / "pair_rotation_local.csl",
    )
    p = dest / "layout.csl"
    text = p.read_text()
    i = text.rfind("}")
    extra = "".join(
        f'@export_name("{name}",[*]{dtype},true);\n' for name, dtype in EXPORTS.items()
    )
    p.write_text(text[:i] + extra + text[i:])
    (dest / "SOURCE-NOTICE.txt").write_text(
        "Source-derived shared-input RMS/QKV/odd_even pair prefix and resident attention/output/FFN tail, MeshInfra/WaferLLM fd1c2daae37cd68706c03fc8009887ecee9900f8 Apache-2.0. Shared gamma and original X residual; forward two-hop QKV projections with preserved live normalized owner, synchronous in-place pair rotation, joined existing CSL engine. No intermediate host transfer. Single-head unmasked supplied broadcast coefficients; not full Prefill/Decode or hardware qualification.\n"
    )
