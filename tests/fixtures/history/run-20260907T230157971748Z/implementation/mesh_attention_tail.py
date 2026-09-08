"""Typed resident supplied-Q/K/V attention followed by projection and FFN tail."""

import copy
from pathlib import Path
import numpy as np
from frontend import check
from input_contracts import effective_bound
from attention_output_bounds import bound as output_bound
from mesh_attention import (
    verify as attention_verify,
    plan as attention_plan,
    evaluate as attention_evaluate,
)
from mesh_prefill_tail import (
    verify as tail_verify,
    plan as tail_plan,
    evaluate as tail_evaluate,
)


def _host(m, stem):
    used = {v.get("host") for v in m["nodes"]}
    while stem in used:
        stem += "_"
    return stem


def attention(m):
    ns = copy.deepcopy(m["nodes"])
    sink = copy.deepcopy(ns[-1])
    sink.update(
        inputs=[ns[12]["id"]],
        shape=ns[12]["shape"][:],
        host=_host(m, "__attention_output"),
    )
    return dict(
        m, nodes=ns[:3] + ns[9:13] + [sink], profile="mesh_attention.v1", input_bound=1
    )


def tail(m):
    ns = copy.deepcopy(m["nodes"])
    a = ns[12]
    b = output_bound(
        ns[0]["shape"][0],
        a["dataflow"]["rows"],
        effective_bound(ns[2], m["input_bound"]),
    )
    a.update(
        op="input",
        inputs=[],
        host=_host(m, "__resident_attention"),
        abs_bound=b["value_output_absolute"],
    )
    a.pop("dataflow", None)
    return dict(m, nodes=[a] + ns[3:9] + ns[13:], profile="mesh_prefill_tail.v1")


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    ns = m["nodes"]
    by = {v["id"]: v for v in ns}
    check(
        len(ns) == len(by) == 23
        and sorted(v["op"] for v in ns)
        == sorted(
            ["input"] * 9
            + ["matmul"] * 6
            + ["add"] * 2
            + ["rmsnorm", "silu", "multiply", "transpose", "softmax", "output"]
        ),
        "23 unique attention-tail nodes",
    )
    check(all(i in by for v in ns for i in v["inputs"]), "defined attention-tail edges")
    sm = next(v for v in ns if v["op"] == "softmax")
    check(len(sm["inputs"]) == 1, "attention softmax operand")
    mm = by[sm["inputs"][0]]
    check(mm["op"] == "matmul" and len(mm["inputs"]) == 2, "attention score product")
    q, t = [by[i] for i in mm["inputs"]]
    check(t["op"] == "transpose" and len(t["inputs"]) == 1, "attention key transpose")
    k = by[t["inputs"][0]]
    consumers = [
        v
        for v in ns
        if v["op"] == "matmul" and len(v["inputs"]) == 2 and v["inputs"][0] == sm["id"]
    ]
    check(len(consumers) == 1, "single resident probability consumer")
    a = consumers[0]
    v = by[a["inputs"][1]]
    check(
        all(x["op"] == "input" for x in (q, k, v))
        and len({x["id"] for x in (q, k, v)}) == 3,
        "distinct supplied Q/K/V",
    )
    check(
        all(effective_bound(x, bound) <= 1 for x in (q, k, v)),
        "attention public Q/K/V bounds",
    )
    sink = copy.deepcopy(next(v for v in ns if v["op"] == "output"))
    sink.update(
        inputs=[a["id"]], host=_host(m, "__attention_output"), shape=a["shape"][:]
    )
    am = attention_verify(dict(m, nodes=[q, k, v, t, mm, sm, a, sink]), epochs, 1)
    b = output_bound(q["shape"][0], a["dataflow"]["rows"], effective_bound(v, bound))
    aa = copy.deepcopy(a)
    aa.update(
        op="input",
        inputs=[],
        host=_host(m, "__resident_attention"),
        abs_bound=b["value_output_absolute"],
    )
    aa.pop("dataflow", None)
    removed = {x["id"] for x in (q, k, v, t, mm, sm)}
    tn = [aa if x["id"] == a["id"] else x for x in ns if x["id"] not in removed]
    tm = tail_verify(dict(m, nodes=tn), epochs, bound)
    canonical = am["nodes"][:3] + tm["nodes"][1:7] + am["nodes"][3:7] + tm["nodes"][7:]
    check(
        len(canonical) == 23 and {x["id"] for x in canonical} == set(by),
        "all attention-tail nodes consumed",
    )
    m.update(
        nodes=canonical,
        profile="mesh_attention_tail.v1",
        epochs=epochs,
        input_bound=bound,
    )
    for x in canonical:
        x["interval"] = None
    plan(m)
    return m


def plan(m, partitions=1):
    a = attention_plan(attention(m), partitions)
    s = tail_plan(tail(m), partitions)
    check(
        all(s[k] == a[k] for k in ("P", "M", "N", "Mt", "Nt", "instrumentation")),
        "one attention-tail region",
    )
    p, l, h = s["P"], s["length"], s["hidden_length"]
    score = s["Mt"] ** 2
    check(score <= h, "score/exponent tiles must fit borrowed up/gate buffers")
    sampled = s["instrumentation"] == "sampled"
    diagnostics = p * (2 * score + 3 * l) + 5 * s["Mt"] + score if sampled else 7
    s["memory_per_pe"]["attention_public_KV_and_observers"] = 2 * (
        3 * l + 2 * score + 2 * s["Mt"] + diagnostics
    ) + 2 * (p + 13)
    s["memory_per_pe"]["attention_code_descriptor_reserve"] = 2048
    check(sum(s["memory_per_pe"].values()) <= 49152, "attention-tail PE memory budget")
    s.update(
        profile="mesh_attention_tail.v1",
        attention_schedule=a,
        input_bindings={
            node["host"]: port
            for node, port in zip(
                m["nodes"][:9],
                (
                    "x",
                    "attention_k",
                    "attention_v",
                    "output_weight",
                    "residual",
                    "gamma",
                    "up_weight",
                    "gate_weight",
                    "down_weight",
                ),
            )
        },
        score_length=score,
        scale=a["scale"],
        attention_output_bound=output_bound(
            s["M"], p, effective_bound(m["nodes"][2], m["input_bound"])
        ),
        composition=dict(
            expression="A=softmax(scale*Q*K^T)*V; Z=A*O+R; Y=Z+MLP(RMSNorm(Z,gamma))",
            intermediate_host_transfer=False,
            phases=dict(score=5, value=4, output_projection=3, up=0, gate=1, down=2),
            entry_reset="contiguous score right DSD; value stride Mt then contiguous output projection",
            borrowed="up/gate: score/exponents then value probability/receive; xwork/xrecv: K then V then tail work",
            completion="single task binding; join each producer before reuse; only final Z residual unblocks host",
        ),
    )
    from attention_tail_lifetimes import plan as lifetimes

    s["storage_lifetimes"] = lifetimes(s)
    s["resources"]["explicit_dsr_phases"]["attention"] = a["resources"]
    return s


def inputs(m, b):
    check(set(b) == {v["host"] for v in m["nodes"][:9]}, "nine attention-tail inputs")
    out = []
    for v in m["nodes"][:9]:
        a = np.asarray(b[v["host"]], float).reshape(v["shape"])
        check(
            np.all(np.isfinite(a))
            and np.all(np.abs(a) <= effective_bound(v, m["input_bound"]))
            and np.array_equal(a, a.astype(np.float16).astype(float)),
            "finite bounded exact-half attention-tail inputs",
        )
        out.append(a)
    return out


def evaluate(m, batches):
    am = attention(m)
    tm = tail(m)
    ab = []
    for b in batches:
        inputs(m, b)
        ab.append({v["host"]: b[v["host"]] for v in am["nodes"][:3]})
    values, _ = attention_evaluate(am, ab)
    tb = [
        dict(
            {v["host"]: b[v["host"]] for v in tm["nodes"][1:7]},
            **{tm["nodes"][0]["host"]: row[am["nodes"][-1]["host"]]},
        )
        for b, row in zip(batches, values)
    ]
    return tail_evaluate(tm, tb)


def generate(s, dest, *, composition_hooks=None):
    from pathlib import Path
    import shutil
    from mesh_prefill_tail import generate as generate_tail
    from attention_tail_csl import hooks, EXPORTS

    def composed(schedule, base):
        configured = hooks(schedule, base)
        return (
            composition_hooks(schedule, configured) if composition_hooks else configured
        )

    generate_tail(s, dest, composition_hooks=composed)
    dest = Path(dest)
    shutil.copyfile(
        Path(__file__).parent / "runtime/softmax_local.csl", dest / "softmax_local.csl"
    )
    p = dest / "layout.csl"
    text = p.read_text()
    i = text.rfind("}")
    extra = "".join(
        f'@export_name("{name}",[*]{dtype},true);\n' for name, dtype in EXPORTS.items()
    )
    p.write_text(text[:i] + extra + text[i:])
    (dest / "SOURCE-NOTICE.txt").write_text(
        "Source-derived supplied-Q/K/V resident attention and output/FFN tail, MeshInfra/WaferLLM fd1c2daae37cd68706c03fc8009887ecee9900f8 Apache-2.0. Rotating-root QK, repaired stable row softmax, column-major aligned PV, output projection/residual/RMS and explicit block-f32 MLP. Shared joined CSL engine, immutable inputs, no intermediate host transfer. Unmasked single head; not full Prefill/Decode or hardware qualification.\n"
    )
