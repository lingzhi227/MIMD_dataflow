"""Typed projection/residual followed by the shared normalized FFN region."""

import copy
from pathlib import Path
import numpy as np
from frontend import check
from binary16 import matmul
from mesh_projection_residual_rms import verify as prefix_verify, plan as prefix_plan
from mesh_feed_forward import (
    verify as ff_verify,
    plan as ff_plan,
    evaluate as ff_evaluate,
)


def prefix(m):
    a, o, r, gamma, u, g, d, mm, z, norm, up, gate, act, hidden, down, add, out = (
        copy.deepcopy(m["nodes"])
    )
    out["inputs"] = [norm["id"]]
    return dict(
        m,
        nodes=[a, o, r, gamma, mm, z, norm, out],
        profile="mesh_projection_residual_rms.v1",
    )


def core(m):
    a, o, r, gamma, u, g, d, mm, z, norm, up, gate, act, hidden, down, add, out = (
        copy.deepcopy(m["nodes"])
    )
    bound = prefix_plan(prefix(m))["numerical_bounds"]["residual_sum"]
    host = "__post_projection_Z"
    while host in {v["host"] for v in (a, o, r, gamma, u, g, d)}:
        host += "_"
    z.update(op="input", inputs=[], host=host, abs_bound=bound)
    z.pop("dataflow", None)
    return dict(
        m,
        nodes=[z, gamma, u, g, d, norm, up, gate, act, hidden, down, add, out],
        profile="mesh_feed_forward.v1",
    )


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    ns = m["nodes"]
    by = {n["id"]: n for n in ns}
    check(
        len(ns) == len(by) == 17
        and sorted(n["op"] for n in ns)
        == sorted(
            ["input"] * 7
            + ["matmul"] * 4
            + ["add"] * 2
            + ["rmsnorm", "silu", "multiply", "output"]
        ),
        "seventeen unique typed tail nodes",
    )
    check(all(i in by for n in ns for i in n["inputs"]), "defined tail edges")

    def operands(n, op, count):
        check(n["op"] == op and len(n["inputs"]) == count, "tail " + op + " edges")
        return [by[i] for i in n["inputs"]]

    out = next(n for n in ns if n["op"] == "output")
    (add,) = operands(out, "output", 1)
    pair = operands(add, "add", 2)
    check(sum(n["op"] == "matmul" for n in pair) == 1, "tail final down projection")
    down = next(n for n in pair if n["op"] == "matmul")
    z = next(n for n in pair if n is not down)
    hidden, d = operands(down, "matmul", 2)
    pair = operands(hidden, "multiply", 2)
    check(sum(n["op"] == "silu" for n in pair) == 1, "tail gating")
    act = next(n for n in pair if n["op"] == "silu")
    up = next(n for n in pair if n is not act)
    (gate,) = operands(act, "silu", 1)
    norm, u = operands(up, "matmul", 2)
    norm2, g = operands(gate, "matmul", 2)
    zz, gamma = operands(norm, "rmsnorm", 2)
    check(
        norm["id"] == norm2["id"] and z["id"] == zz["id"],
        "final residual retains postprojection Z",
    )
    pair = operands(z, "add", 2)
    check(sum(n["op"] == "matmul" for n in pair) == 1, "tail output projection")
    mm = next(n for n in pair if n["op"] == "matmul")
    r = next(n for n in pair if n is not mm)
    a, o = operands(mm, "matmul", 2)
    sources = [a, o, r, gamma, u, g, d]
    check(
        all(n["op"] == "input" and not n["inputs"] for n in sources)
        and len({n["id"] for n in sources}) == len({n["host"] for n in sources}) == 7,
        "seven distinct tail inputs",
    )
    canonical = sources + [mm, z, norm, up, gate, act, hidden, down, add, out]
    check({n["id"] for n in canonical} == set(by), "all tail nodes consumed")
    m.update(
        nodes=canonical,
        profile="mesh_prefill_tail.v1",
        epochs=epochs,
        input_bound=bound,
    )
    prefix_verify(prefix(m), epochs, bound)
    ff_verify(core(m), epochs, bound)
    out.update(shape=z["shape"][:], dtype="f16")
    for n in canonical:
        n["interval"] = None
    plan(m)
    return m


def plan(m, partitions=1):
    pre = prefix_plan(prefix(m), partitions)
    s = ff_plan(core(m), partitions)
    check(pre["P"] == s["P"] and pre["length"] == s["length"], "one tail region")
    l = s["length"]
    ow = s["Nt"] ** 2
    sample = s["instrumentation"] == "sampled"
    s["memory_per_pe"]["projection_prelude_inputs_live_Z_and_observers"] = (
        2 * (3 * l + ow + (s["P"] * l + l + ow if sample else 3)) + 8
    )
    check(sum(s["memory_per_pe"].values()) <= 49152, "tail PE memory budget")
    s.update(
        profile="mesh_prefill_tail.v1",
        projection_prelude=pre["projection_stage"],
        prelude_numerical_bounds=pre["numerical_bounds"],
        output_weight_length=ow,
        input_bindings={
            m["nodes"][0]["host"]: "x",
            m["nodes"][1]["host"]: "output_weight",
            m["nodes"][2]["host"]: "residual",
            m["nodes"][3]["host"]: "gamma",
        },
        composition=dict(
            expression="Z=attention*output_weight+residual; Y=Z+MLP(RMSNorm(Z,gamma))",
            intermediate_host_transfer=False,
            prelude_phase=3,
            core_phases=[0, 1, 2],
            residual="post_projection_z remains live through final add",
            completion="joined projection prelude -> synchronous residual/RMS -> shared joined MLP",
        ),
    )
    from prefill_tail_lifetimes import plan as lifetimes

    s["storage_lifetimes"] = lifetimes(s)
    s["resources"]["explicit_dsr_phases"]["projection_prelude"] = s["resources"][
        "explicit_dsr_lease"
    ]
    return s


def inputs(m, b):
    from input_contracts import effective_bound

    sources = m["nodes"][:7]
    check(set(b) == {n["host"] for n in sources}, "tail input ports")
    out = []
    for n in sources:
        v = np.asarray(b[n["host"]], float).reshape(n["shape"])
        check(
            np.all(np.isfinite(v))
            and np.all(np.abs(v) <= effective_bound(n, m["input_bound"]))
            and np.array_equal(v, v.astype(np.float16).astype(float)),
            "finite bounded exact-half tail inputs",
        )
        out.append(v)
    return out


def evaluate(m, batches):
    c = core(m)
    cb = []
    for b in batches:
        a, o, r, gamma, u, g, d = inputs(m, b)
        z = (matmul(a, o) + r).astype(np.float16).astype(float)
        cb.append(
            {
                n["host"]: v.ravel().tolist()
                for n, v in zip(c["nodes"][:5], (z, gamma, u, g, d))
            }
        )
    return ff_evaluate(c, cb)


def generate(s, dest):
    from mesh_feed_forward import generate as generate_ff
    from prefill_tail_csl import hooks

    generate_ff(s, dest, composition_hooks=hooks, residual_buffer="post_projection_z")
    dest = Path(dest)
    p = dest / "layout.csl"
    text = p.read_text()
    i = text.rfind("}")
    extra = "".join(
        f'@export_name("{name}",[*]{dtype},true);\n'
        for name, dtype in [
            ("output_weight", "f16"),
            ("residual", "f16"),
            ("projection_snapshot", "f16"),
            ("post_projection_z", "f16"),
            ("projection_history", "f16"),
            ("projection_left_first", "f16"),
            ("projection_right_first", "f16"),
            ("prelude_progress", "u16"),
        ]
    )
    p.write_text(text[:i] + extra + text[i:])
    (dest / "SOURCE-NOTICE.txt").write_text(
        "Source-derived supplied-attention-output tail, MeshInfra/WaferLLM fd1c2daae37cd68706c03fc8009887ecee9900f8 Apache-2.0. Original h1/z_add -> repaired RMS/row collective -> shared all-block MLP/gate ownership repair -> final postprojection Z add. Shared CSL projection engine with joined typed prelude; immutable public inputs. Not full Prefill/Decode or hardware qualification.\n"
    )
