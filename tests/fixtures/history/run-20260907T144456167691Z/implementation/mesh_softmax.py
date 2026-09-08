"""Stable distributed row softmax with explicit SDK half arithmetic policy."""

import copy, math
from pathlib import Path
import numpy as np
from frontend import check

POLICY = dict(
    partition="tiles",
    reduce="max_sum",
    accumulation="f16",
    math="sdk_half",
    compute="dsr",
    fp="relaxed",
)


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    check(
        [n["op"] for n in m["nodes"]] == ["input", "softmax", "output"],
        "softmax typed graph",
    )
    x, op, out = m["nodes"]
    check("dataflow" in op, "softmax requires an explicit dataflow policy")
    check(
        len({n["id"] for n in m["nodes"]}) == 3
        and op["inputs"] == [x["id"]]
        and out["inputs"] == [op["id"]],
        "softmax dependencies",
    )
    check(
        not m["states"] and not any("place" in n for n in m["nodes"]),
        "softmax owned region",
    )
    check(
        x.get("dtype") == op.get("dtype") == "f16" and x["shape"] == op["shape"],
        "softmax half shapes",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 4096,
        "softmax execution bounds",
    )
    check(
        math.isfinite(op["scale"]) and 0 < float(np.float16(op["scale"])) <= 8,
        "softmax finite positive scale",
    )
    check(
        2 * bound * float(np.float16(op["scale"])) <= 65504,
        "softmax half scaled-difference bound",
    )
    out.update(shape=op["shape"][:], dtype="f16")
    for n in m["nodes"]:
        n["interval"] = None
    m.update(profile="mesh_softmax.v1", epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    x, op, out = m["nodes"]
    d = op["dataflow"]
    rows, cols = d["rows"], d["cols"]
    M, N = x["shape"]
    check(
        set(d)
        in (
            set(POLICY) | {"rows", "cols"},
            set(POLICY) | {"rows", "cols", "elementwise"},
        )
        and d.get("elementwise", "scalar") in ("map", "scalar")
        and all(d[k] == v for k, v in POLICY.items()),
        "softmax complete policy",
    )
    check(
        partitions == 1
        and type(rows) is int
        and type(cols) is int
        and rows in (2, 4, 8, 16)
        and cols in (4, 8, 16),
        "softmax bounded region",
    )
    check(
        type(M) is int
        and type(N) is int
        and 1 <= M <= 512
        and 4 <= N <= 2048
        and M % rows == N % cols == 0,
        "softmax divisible tiles",
    )
    mt, nt = M // rows, N // cols
    mode = m.get("instrumentation", "sampled")
    check(mode in ("sampled", "counters"), "softmax instrumentation")
    memory = dict(
        data=6 * mt * nt + 4 * mt,
        observations=10 * mt if mode == "sampled" else 2,
        protocol=128,
        code_stack_reserve=16384,
    )
    check(
        mt * nt <= 32767 and sum(memory.values()) <= 49152,
        "softmax descriptor/SRAM limits",
    )
    schedule = dict(
        profile=m["profile"],
        rows=rows,
        cols=cols,
        M=M,
        N=N,
        Mt=mt,
        Nt=nt,
        scale=op["scale"],
        instrumentation=mode,
        epochs=m["epochs"],
        dtype="f16",
        memory_per_pe=memory,
        stages=[
            dict(kind="scale_local_max", initializer="negative_max_finite"),
            dict(kind="row_allreduce", combine="max"),
            dict(kind="max_shift_sdk_exp_local_sum"),
            dict(kind="row_allreduce", combine="add", root_order=["east", "west"]),
            dict(kind="reciprocal"),
            dict(kind="row_vector_scale"),
        ],
        resources=dict(
            colors=list(range(4, 9)),
            input_queues=[3, 4, 6],
            output_queues=[3, 4, 6],
            compute_dsr=1,
            reduction_src1_dsr=2,
            local_tasks=[],
            host_bindings="fixed input/result/exponent storage; one host invocation",
        ),
        nodes=[
            dict(id=f"p{c}_{r}", tile=[c, r], place=[4 + c, 1 + r])
            for r in range(rows)
            for c in range(cols)
        ],
    )

    if "elementwise" in d:
        schedule["elementwise"] = d["elementwise"]
        schedule["stages"][2]["elementwise"] = d["elementwise"]
        schedule["resources"][
            "elementwise_descriptors"
        ] = "compiler-managed synchronous phase; no sharing with simultaneous work"
    return schedule


def inputs(m, b):
    x = m["nodes"][0]
    check(set(b) == {x["host"]}, "softmax input port")
    check(
        all(type(v) in (int, float) for v in b[x["host"]]), "softmax scalar input types"
    )
    a = np.asarray(b[x["host"]], float)
    check(
        a.shape == (math.prod(x["shape"]),)
        and np.all(np.isfinite(a))
        and np.all(np.abs(a) <= m["input_bound"]),
        "softmax input extent/bounds",
    )
    check(
        np.array_equal(a, a.astype(np.float16).astype(float)),
        "softmax exact half input",
    )
    return a.reshape(x["shape"])


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "softmax epoch count")
    outputs = []
    for b in batches:
        x = inputs(m, b) * m["nodes"][1]["scale"]
        v = np.exp(x - np.max(x, axis=1)[:, None])
        v /= np.sum(v, axis=1)[:, None]
        outputs.append(
            {m["nodes"][2]["host"]: v.astype(np.float16).astype(float).ravel().tolist()}
        )
    return outputs, {}


def reference(s, x):
    # The exp model must be independently qualified before using this backend.
    from sdk_math_reference import exp_f16_nonpositive

    q = lambda a: np.asarray(a, np.float16).astype(float)
    rows, cols, mt, nt = s["rows"], s["cols"], s["Mt"], s["Nt"]
    scaled = q(x * float(np.float16(s["scale"])))
    check(np.all(np.isfinite(scaled)), "softmax scaled half overflow")
    local_max = scaled.reshape(rows, mt, cols, nt).transpose(0, 2, 1, 3).max(axis=3)
    maximum = local_max.max(axis=1)
    global_max = np.repeat(maximum[:, None, :], cols, axis=1)
    shifted = q(scaled - maximum.reshape(s["M"], 1))
    check(np.all(np.isfinite(shifted)), "softmax half shift overflow")
    exponent = np.asarray([exp_f16_nonpositive(v) for v in shifted.ravel()]).reshape(
        x.shape
    )
    local_sum = np.zeros((rows, cols, mt))
    for c in range(cols):
        for j in range(nt):
            local_sum[:, c] = q(
                local_sum[:, c] + exponent[:, c * nt + j].reshape(rows, mt)
            )
    left = local_sum[:, 0].copy()
    for c in range(1, cols // 2):
        left = q(left + local_sum[:, c])
    right = local_sum[:, -1].copy()
    for c in range(cols - 2, cols // 2, -1):
        right = q(right + local_sum[:, c])
    total = q(q(local_sum[:, cols // 2] + right) + left)
    check(np.all((total > 0) & np.isfinite(total)), "softmax positive finite row sums")
    inverse = q(1 / total)
    output = q(exponent * inverse.reshape(s["M"], 1))
    history = np.stack(
        [
            local_max,
            global_max,
            local_sum,
            np.repeat(total[:, None, :], cols, axis=1),
            np.repeat(inverse[:, None, :], cols, axis=1),
        ],
        axis=2,
    )
    return history, exponent, output


def source_text(s, name):
    source = (
        Path(__file__).parent / "runtime" / ("softmax_" + name + ".csl")
    ).read_text()
    if name == "pe" and s.get("elementwise", "scalar") == "map":
        loop = " for(@range(i16,Mt*Nt)) |i| {exponents[i]=math.exp_f16(result[i]);}"
        check(source.count(loop) == 1, "softmax elementwise lowering site")
        source = source.replace(loop, " @map(hls_exp_value,yd,hls_exp_dsd);")
        source += "\nconst hls_exp_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{Mt*Nt}->exponents[i]});\nfn hls_exp_value(v:f16) f16 {return math.exp_f16(v);}\n"
    return source


def generate(s, dest):
    for name in ("layout", "pe"):
        (Path(dest) / (name + ".csl")).write_text(source_text(s, name))
    (Path(dest) / "row_chain.csl").write_text(
        (Path(__file__).parent / "runtime/row_chain.csl").read_text()
    )
