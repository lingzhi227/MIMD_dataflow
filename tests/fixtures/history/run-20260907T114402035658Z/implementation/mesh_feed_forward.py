"""Typed RMS -> gated MLP -> residual, sharing the existing CSL projection engine."""

import copy, math
from pathlib import Path
import numpy as np
from frontend import check
from input_contracts import effective_bound
from mesh_rms import verify as rms_verify, reference as rms_reference
from mesh_mlp import verify_graph, plan as mlp_plan, evaluate as mlp_evaluate
from rms_bounds import row_norm_bound
from binary16 import quantize, bits
from mesh_projection_residual_rms import ADD_POLICY


def norm_bounds(m):
    z, gamma, *_ = m["nodes"]
    norm = m["nodes"][5]
    p = norm["dataflow"]["cols"]
    return row_norm_bound(
        effective_bound(z, m["input_bound"]),
        effective_bound(gamma, m["input_bound"]),
        z["shape"][1] // p,
        p,
        norm["epsilon"],
    )


def core(m):
    z, gamma, wu, wg, wd, norm, up, gate, act, hidden, down, add, out = copy.deepcopy(
        m["nodes"]
    )
    b = norm_bounds(m)["output"]
    host = "__normalized"
    while host in {n["host"] for n in (z, gamma, wu, wg, wd)}:
        host += "_"
    norm.update(op="input", inputs=[], host=host, abs_bound=b)
    norm.pop("dataflow", None)
    norm.pop("epsilon", None)
    out["inputs"] = [down["id"]]
    return dict(
        m,
        nodes=[norm, wu, wg, wd, up, gate, act, hidden, down, out],
        profile="mesh_mlp.v1",
        input_bound=max(m["input_bound"], math.ceil(b)),
    )


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    ns = m["nodes"]
    by = {n["id"]: n for n in ns}
    check(
        len(by) == len(ns) == 13
        and sorted(n["op"] for n in ns)
        == sorted(
            ["input"] * 5
            + ["matmul"] * 3
            + ["rmsnorm", "silu", "multiply", "add", "output"]
        ),
        "feed-forward thirteen unique typed nodes",
    )
    check(all(i in by for n in ns for i in n["inputs"]), "defined feed-forward edges")

    def operands(n, op, count):
        check(
            n["op"] == op and len(n["inputs"]) == count, "feed-forward " + op + " edges"
        )
        return [by[i] for i in n["inputs"]]

    out = next(n for n in ns if n["op"] == "output")
    (add,) = operands(out, "output", 1)
    pair = operands(add, "add", 2)
    check(
        sum(n["op"] == "matmul" for n in pair) == 1,
        "final residual consumes the down projection",
    )
    down = next(n for n in pair if n["op"] == "matmul")
    z = next(n for n in pair if n is not down)
    hidden, wd = operands(down, "matmul", 2)
    pair = operands(hidden, "multiply", 2)
    check(sum(n["op"] == "silu" for n in pair) == 1, "feed-forward gating")
    act = next(n for n in pair if n["op"] == "silu")
    up = next(n for n in pair if n is not act)
    (gate,) = operands(act, "silu", 1)
    norm, wu = operands(up, "matmul", 2)
    norm2, wg = operands(gate, "matmul", 2)
    zz, gamma = operands(norm, "rmsnorm", 2)
    check(
        norm["id"] == norm2["id"] and z["id"] == zz["id"],
        "RMS input is the final residual; shared normalized activation",
    )
    sources = [z, gamma, wu, wg, wd]
    check(
        all(n["op"] == "input" and not n["inputs"] for n in sources)
        and len({n["id"] for n in sources}) == len({n["host"] for n in sources}) == 5,
        "five distinct feed-forward inputs",
    )
    check(
        not m["states"] and not any("place" in n for n in ns),
        "feed-forward owns its resident region",
    )
    check(
        all(n.get("dtype") == "f16" for n in ns if n is not out),
        "half feed-forward graph",
    )
    check(
        type(bound) is int
        and 0 <= bound <= 2
        and type(epochs) is int
        and 1 <= epochs <= 16,
        "feed-forward execution bounds",
    )
    check(add["shape"] == down["shape"] == z["shape"], "final residual shape")
    rm = dict(m, nodes=[z, gamma, norm, dict(out, inputs=[norm["id"]])])
    rms_verify(rm, epochs, bound)
    p = norm["dataflow"]["cols"]
    check(
        norm["dataflow"]["rows"] == p and p in (4, 8), "one square feed-forward region"
    )
    check(
        add.get("dataflow") == dict(ADD_POLICY, rows=p, cols=p),
        "explicit final residual policy",
    )
    m["nodes"] = sources + [norm, up, gate, act, hidden, down, add, out]
    out.update(shape=z["shape"][:], dtype="f16")
    for n in m["nodes"]:
        n["interval"] = None
    m.update(profile="mesh_feed_forward.v1", epochs=epochs, input_bound=bound)
    c = core(m)
    verify_graph(c, epochs, c["input_bound"])
    plan(m)
    return m


def plan(m, partitions=1):
    from feed_forward_lifetimes import plan as lifetime_plan
    from inference_resources import compute, registers, row_reduce, projection_lease

    s = mlp_plan(core(m), partitions)
    p = s["P"]
    norm = m["nodes"][5]
    check(
        norm["dataflow"]["rows"] == norm["dataflow"]["cols"] == p,
        "RMS and MLP share the region",
    )
    check(
        s["M"] <= 128 and s["N"] <= 128 and s.get("down_accumulation") == "block_f32",
        "first normalized feed-forward uses bounded block-f32 down",
    )
    b = norm_bounds(m)
    zbound = effective_bound(m["nodes"][0], m["input_bound"])
    check(
        s["numerical_bounds"]["output"] + zbound <= 65504, "final residual half range"
    )
    s["memory_per_pe"]["normalization_and_delta_observers"] = (
        2 * (s["Nt"] + s["Mt"] + 2 * s["length"]) + 4
    )
    check(sum(s["memory_per_pe"].values()) <= 49152, "feed-forward PE memory budget")
    s.update(
        profile=m["profile"],
        epsilon=norm["epsilon"],
        normalization_bounds=b,
        input_bindings={m["nodes"][0]["host"]: "x", m["nodes"][1]["host"]: "gamma"},
        composition=dict(
            expression="Z + MLP(RMSNorm(Z,gamma))",
            intermediate_host_transfer=False,
            residual="immutable public x storage retains Z through all MLP stages",
            normalized="private xwork, consumed by both up/gate, then reused for down/final result",
            scratch="xrecv: local square scratch, then joined projection receive buffer",
            delta_observer="actual narrowed down result copied before residual addition; observed in both modes",
            completion="RMS local/row collective/inverse/normalize returns before MLP entry; joined MLP finish precedes residual add",
        ),
    )
    s["storage_lifetimes"] = lifetime_plan(s)
    s["resources"]["explicit_dsr_phases"] = dict(
        local_square_sum=compute() + registers(2, "dest", "src0", "src1"),
        row_collective=row_reduce(),
        inverse="SDK-managed temporaries; no explicit user DSR lease",
        normalize=compute(),
        projections=projection_lease(),
        gating=compute(),
        final_residual=compute(),
    )
    return s


def inputs(m, b):
    sources = m["nodes"][:5]
    check(set(b) == {n["host"] for n in sources}, "feed-forward input ports")
    arrays = []
    for n in sources:
        a = np.asarray(b[n["host"]], float).reshape(n["shape"])
        check(
            np.all(np.isfinite(a))
            and np.all(np.abs(a) <= effective_bound(n, m["input_bound"]))
            and np.array_equal(a, a.astype(np.float16).astype(float)),
            "finite bounded exact-half feed-forward inputs",
        )
        arrays.append(a)
    return arrays


def normalized(s, z, gamma, target=True):
    if target:
        return rms_reference(s, z, gamma)[-1]
    return (
        (z * gamma / np.sqrt(np.mean(z * z, axis=1)[:, None] + s["epsilon"]))
        .astype(np.float16)
        .astype(float)
    )


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "feed-forward epochs")
    c = core(m)
    s = plan(m)
    cb = []
    zs = []
    for b in batches:
        z, gamma, u, g, d = inputs(m, b)
        zs.append(z)
        cb.append(
            {
                n["host"]: a.ravel().tolist()
                for n, a in zip(
                    c["nodes"][:4], (normalized(s, z, gamma, False), u, g, d)
                )
            }
        )
    outputs, _ = mlp_evaluate(c, cb)
    name = m["nodes"][-1]["host"]
    return [
        {
            name: (z + np.asarray(o[name]).reshape(z.shape))
            .astype(np.float16)
            .astype(float)
            .ravel()
            .tolist()
        }
        for z, o in zip(zs, outputs)
    ], {}


def generate(s, dest, *, composition_hooks=None, residual_buffer=None):
    from feed_forward_csl import hooks
    from mesh_mlp import generate as generate_mlp

    dest = Path(dest)
    configured = hooks(bits(quantize(s["epsilon"])), residual_buffer)
    if composition_hooks is not None:
        configured = composition_hooks(s, configured)
    generate_mlp(s, dest, hooks=configured)
    rt = Path(__file__).parent / "runtime"
    (dest / "rms_local.csl").write_bytes((rt / "rms_local.csl").read_bytes())
    layout = dest / "layout.csl"
    text = layout.read_text()
    i = text.rfind("}")
    extra = "".join(
        f'@export_name("{name}",[*]{dtype},true);\n'
        for name, dtype in (
            ("gamma", "f16"),
            ("normalized", "f16"),
            ("down_snapshot", "f16"),
            ("rms_progress", "u16"),
        )
    )
    layout.write_text(text[:i] + extra + text[i:])
    (dest / "SOURCE-NOTICE.txt").write_text(
        "Source-derived normalized feed-forward residual; MeshInfra/WaferLLM fd1c2daae37cd68706c03fc8009887ecee9900f8 Apache-2.0. Repaired RMS local library with original row collective, shared MLP engine/gate carry repair, explicit block-f32 down accumulation, actual down snapshot then original Z residual. Immutable public inputs; no intermediate host transfers. Supplied Z/gamma/U/G/D, not full Prefill/Decode or hardware qualification.\n"
    )
