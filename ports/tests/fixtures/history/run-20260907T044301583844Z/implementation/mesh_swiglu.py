"""Typed pointwise SiLU/gating subgraph with bounded SDK half math semantics."""

import copy
from pathlib import Path
import numpy as np
from frontend import check
from sdk_math_reference import silu_f16

POLICY = dict(
    partition="tiles", math="sdk_half", compute="dsr", elementwise="map", fp="relaxed"
)


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    nodes = m["nodes"]
    check(
        len(nodes) == 5 and len({n["id"] for n in nodes}) == 5,
        "gated activation five unique nodes",
    )
    check(
        sorted(n["op"] for n in nodes)
        == ["input", "input", "multiply", "output", "silu"],
        "gated activation typed subgraph",
    )
    by = {n["id"]: n for n in nodes}
    act = next(n for n in nodes if n["op"] == "silu")
    mul = next(n for n in nodes if n["op"] == "multiply")
    out = next(n for n in nodes if n["op"] == "output")
    check(
        len(act["inputs"]) == 1
        and len(mul["inputs"]) == 2
        and all(v in by for n in nodes for v in n["inputs"]),
        "gated activation defined operands",
    )
    gate = by[act["inputs"][0]]
    check(mul["inputs"].count(act["id"]) == 1, "gated activation last-use edge")
    up = by[next(v for v in mul["inputs"] if v != act["id"])]
    check(
        up["op"] == gate["op"] == "input"
        and up["id"] != gate["id"]
        and up["host"] != gate["host"]
        and not up["inputs"]
        and not gate["inputs"],
        "distinct gating source tensors",
    )
    check(out["inputs"] == [mul["id"]], "gated activation output edge")
    check(
        not m["states"] and not any("place" in n for n in nodes),
        "gated activation owns region",
    )
    check(
        all(n.get("dtype") == "f16" for n in (up, gate, act, mul)),
        "gated activation binary16 tensors",
    )
    shape = up["shape"]
    check(
        len(shape) == 2 and all(type(v) is int and 1 <= v <= 4096 for v in shape),
        "gated activation bounded matrix shape",
    )
    check(
        all(n["shape"] == shape for n in (gate, act, mul)),
        "gated activation equal shapes",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 8,
        "gated activation finite input magnitude bound0..8",
    )
    check(
        "dataflow" in act and act["dataflow"] == mul.get("dataflow"),
        "gated activation shared explicit policy",
    )
    out.update(shape=shape[:], dtype="f16")
    m["nodes"] = [up, gate, act, mul, out]
    for n in nodes:
        n["interval"] = None
    m.update(profile="mesh_swiglu.v1", epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    up, gate, act, mul, out = m["nodes"]
    d = act["dataflow"]
    check(
        set(d) == set(POLICY) | {"rows", "cols"}
        and all(d[k] == v for k, v in POLICY.items()),
        "gated activation dataflow policy",
    )
    rows, cols = d["rows"], d["cols"]
    check(
        partitions == 1 and all(type(v) is int and 1 <= v <= 16 for v in (rows, cols)),
        "gated activation region bounds",
    )
    M, N = up["shape"]
    check(M % rows == 0 and N % cols == 0, "gated activation divisible tiles")
    mt, nt = M // rows, N // cols
    length = mt * nt
    check(length <= 32767, "gated activation signed DSD length")
    mode = m.get("instrumentation", "sampled")
    check(mode in ("sampled", "counters"), "gated activation observation mode")
    memory = dict(
        input_and_work_bytes=6 * length,
        activation_observation_bytes=2 * length if mode == "sampled" else 2,
        control_and_descriptors_reserve=1024,
        sdk_code_stack_reserve=8192,
    )
    check(sum(memory.values()) <= 49152, "gated activation PE memory budget")
    return dict(
        profile="mesh_swiglu.v1",
        M=M,
        N=N,
        rows=rows,
        cols=cols,
        Mt=mt,
        Nt=nt,
        length=length,
        epochs=m["epochs"],
        instrumentation=mode,
        memory_per_pe=memory,
        stages=[
            dict(
                id=0, op="silu", compute="synchronous CSL @map; official math.exp_f16"
            ),
            dict(
                id=1,
                op="multiply",
                compute="DSR fmulh; reuse activation buffer after its last read",
            ),
        ],
        resources=dict(
            colors=[],
            input_queues=[],
            output_queues=[],
            local_tasks=[],
            microthreads=[],
            explicit_compute_dsr=1,
            ownership="SDK memcpy owns I/O and launch machinery; no user fabric communication or asynchronous tasks; compiler-managed map temporaries are phase-exclusive",
        ),
        storage=dict(
            up="immutable host input",
            gate="immutable host input",
            result="activation temporary then gated output",
            order="column-major",
        ),
        nodes=[
            dict(id=f"p{x}_{y}", tile=[x, y], place=[4 + x, 1 + y])
            for y in range(rows)
            for x in range(cols)
        ],
    )


def inputs(m, b):
    nodes = m["nodes"][:2]
    check(set(b) == {n["host"] for n in nodes}, "gated activation input ports")
    arrays = []
    for n in nodes:
        v = b[n["host"]]
        check(
            len(v) == np.prod(n["shape"]) and all(type(x) in (int, float) for x in v),
            "gating input extent/type",
        )
        a = np.asarray(v, float).reshape(n["shape"])
        check(
            np.all(np.isfinite(a)) and np.all(np.abs(a) <= m["input_bound"]),
            "gating finite input bounds",
        )
        check(
            np.array_equal(a, a.astype(np.float16).astype(float)),
            "gating exactly representable half inputs",
        )
        arrays.append(a)
    return arrays


def standard(up, gate):
    e = np.exp(-np.abs(gate))
    activation = np.where(gate >= 0, gate / (1 + e), gate * e / (1 + e))
    return activation, up * activation


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "gated activation epoch count")
    out = []
    for b in batches:
        up, gate = inputs(m, b)
        act = standard(up, gate)[0].astype(np.float16).astype(float)
        out.append(
            {
                m["nodes"][-1]["host"]: (up * act)
                .astype(np.float16)
                .astype(float)
                .ravel()
                .tolist()
            }
        )
    return out, {}


def reference(up, gate):
    activation = np.asarray([silu_f16(v) for v in gate.ravel()]).reshape(gate.shape)
    return activation, (up * activation).astype(np.float16).astype(float)


def accuracy(up, gate, actual):
    nominal = standard(up, gate)[1]
    error = np.abs(actual - nominal)
    allowance = 0.004 * np.abs(nominal) + 2**-24 * (1 + np.abs(up))
    check(
        np.all(np.isfinite(actual)) and np.all(error <= allowance),
        "gated activation half componentwise accuracy",
    )
    return dict(
        contract="gated-activation-half-v1",
        fixed_accuracy_passed=True,
        max_abs_error=float(error.max()),
        max_error_over_allowance=float(np.max(error / allowance)),
        componentwise_relative_term=0.004,
        absolute_rounding_term="2^-24*(1+abs(up))",
        standard_reference="unrounded stable mathematical SiLU times up; allowance includes stored half activation and final half product",
    )


def generate(s, dest):
    rt = Path(__file__).parent / "runtime"
    for name in ("layout", "pe"):
        (Path(dest) / (name + ".csl")).write_text(
            (rt / ("swiglu_" + name + ".csl")).read_text()
        )
    (Path(dest) / "WaferLLM-LICENSE.txt").write_bytes(
        (rt / "waferllm-LICENSE.txt").read_bytes()
    )
    (Path(dest) / "SOURCE-NOTICE.txt").write_text(
        "Gated activation source: MeshInfra/WaferLLM Prefill silu_kernel/z3_comp, commit fd1c2daae37cd68706c03fc8009887ecee9900f8, Apache-2.0. Changes: parameterized region and tile size; activation buffer reused for product; optional observation; stable host inputs and phase diagnostics. No full MLP/inference claim.\n"
    )
