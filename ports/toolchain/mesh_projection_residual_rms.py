"""Typed projection -> residual add -> row RMS, with explicit spatial lifetimes.

Development lowering; dispatch is connected only with its backend/SDK path.
"""

import copy, math
from pathlib import Path
import numpy as np
from frontend import check
from input_contracts import effective_bound, half_dot_bound
from binary16 import matmul, quantize
from projection_contract import stage as projection_stage
from mesh_rms import verify as verify_rms, reference as rms_reference
from rms_bounds import row_norm_bound
from inference_resources import projection_lease, row_reduce, compute, registers

ADD_POLICY = dict(partition="tiles", compute="dsr", fp="relaxed")


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    ns = m["nodes"]
    by = {n["id"]: n for n in ns}
    check(
        len(ns) == len(by) == 8
        and sorted(n["op"] for n in ns)
        == sorted(["input"] * 4 + ["matmul", "add", "rmsnorm", "output"]),
        "projection/residual/RMS eight unique typed nodes",
    )
    check(
        all(i in by for n in ns for i in n["inputs"]),
        "projection/residual/RMS defined edges",
    )

    def operands(n, op, count):
        check(
            n["op"] == op and len(n["inputs"]) == count,
            "projection/residual/RMS " + op + " edges",
        )
        return [by[i] for i in n["inputs"]]

    out = next(n for n in ns if n["op"] == "output")
    (norm,) = operands(out, "output", 1)
    summed, gamma = operands(norm, "rmsnorm", 2)
    pair = operands(summed, "add", 2)
    check(
        sum(n["op"] == "matmul" for n in pair) == 1,
        "residual add consumes one projection",
    )
    mm = next(n for n in pair if n["op"] == "matmul")
    residual = next(n for n in pair if n is not mm)
    activation, weight = operands(mm, "matmul", 2)
    sources = [activation, weight, residual, gamma]
    check(
        all(n["op"] == "input" and not n["inputs"] for n in sources)
        and len({n["id"] for n in sources}) == 4,
        "four independent supplied tensors",
    )
    check(len({n["host"] for n in sources}) == 4, "unique input host ports")
    check(
        not m["states"] and not any("place" in n for n in ns),
        "composition owns its region",
    )
    check(
        all(n.get("dtype") == "f16" for n in ns if n["op"] != "output"),
        "half typed composition",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 2,
        "composition execution bounds",
    )
    check(
        not any(k in mm for k in ("accumulation", "block_size"))
        and "accumulation" not in mm["dataflow"],
        "this projection currently uses source half accumulation",
    )
    projection = projection_stage(activation["shape"], weight["shape"], mm["dataflow"])
    M, N = activation["shape"]
    P = projection["P"]
    check(
        weight["shape"] == [N, N]
        and mm["shape"]
        == residual["shape"]
        == summed["shape"]
        == norm["shape"]
        == [M, N]
        and gamma["shape"] == [1, N],
        "source square projection and row normalization shapes",
    )
    check(P in (4, 8) and M <= 128 and N <= 256, "bounded source region and dimensions")
    check(
        summed.get("dataflow") == dict(ADD_POLICY, rows=P, cols=P),
        "explicit local residual policy on the same region",
    )
    synthetic = copy.deepcopy(m)
    internal_host = "__residual_sum"
    while internal_host in {n["host"] for n in sources}:
        internal_host += "_"
    placeholder = dict(summed, op="input", inputs=[], host=internal_host)
    placeholder.pop("dataflow", None)
    synthetic["nodes"] = [placeholder, gamma, norm, out]
    verify_rms(synthetic, epochs, bound)
    check(
        norm["dataflow"]["rows"] == norm["dataflow"]["cols"] == P,
        "projection and RMS share a region",
    )
    m["nodes"] = sources + [mm, summed, norm, out]
    out.update(shape=[M, N], dtype="f16")
    for n in m["nodes"]:
        n["interval"] = None
    m.update(
        profile="mesh_projection_residual_rms.v1", epochs=epochs, input_bound=bound
    )
    plan(m)
    return m


def plan(m, partitions=1):
    check(partitions == 1, "one resident region")
    x, w, r, g, mm, add, norm, out = m["nodes"]
    projection = projection_stage(x["shape"], w["shape"], mm["dataflow"])
    M, N = x["shape"]
    p = projection["P"]
    mt = M // p
    nt = N // p
    l = mt * nt
    q = nt * nt
    check(l % 4 == q % 4 == 0, "source four-half communication packing")
    mode = m.get("instrumentation", "sampled")
    check(mode in ("sampled", "counters"), "instrumentation mode")
    projection_bound = half_dot_bound(
        effective_bound(x, m["input_bound"]), effective_bound(w, m["input_bound"]), N
    )
    combined = projection_bound + effective_bound(r, m["input_bound"])
    check(combined <= 65504, "residual half addition may overflow")
    zbound = quantize(combined)
    bounds = row_norm_bound(
        zbound, effective_bound(g, m["input_bound"]), nt, p, norm["epsilon"]
    )
    memory = dict(
        public_half_inputs=2 * (2 * l + q + nt),
        private_half_work_and_result=2 * (3 * l + 2 * q + mt),
        observation_half_arrays=2
        * ((p + 2) * l + q + 2 * mt if mode == "sampled" else 6),
        descriptor_and_protocol_reserve=2048,
        sdk_code_tasks_stack_reserve=16384,
    )
    check(sum(memory.values()) <= 49152, "projection/residual/RMS PE memory budget")
    return dict(
        profile=m["profile"],
        M=M,
        N=N,
        P=p,
        rows=p,
        cols=p,
        Mt=mt,
        Nt=nt,
        length=l,
        weight_length=q,
        epsilon=norm["epsilon"],
        epochs=m["epochs"],
        instrumentation=mode,
        projection_stage=projection,
        memory_per_pe=memory,
        numerical_bounds=dict(
            projection=projection_bound, residual_sum=zbound, normalization=bounds
        ),
        stages=[
            dict(
                operation="matmul",
                rounds=p,
                completion="all compute and both communication axes join",
            ),
            dict(
                operation="add",
                completion="synchronous half vector add in logical tile order",
            ),
            dict(
                operation="rmsnorm",
                completion="local squares, synchronous row collective, inverse and row-vector scale",
            ),
        ],
        ownership=dict(
            public_inputs="immutable activation/weight/residual/gamma",
            post_projection_sum="dead private activation buffer",
            square_scratch="dead left receive buffer",
            normalized_output="projection result after residual sum consumes it",
            intermediate_host_transfer=False,
        ),
        resources=dict(
            colors=list(range(1, 12)),
            input_queues=list(range(3, 8)),
            output_queues=list(range(3, 8)),
            local_tasks=[19, 20, 25, 26],
            microthreads=list(range(4)),
            explicit_dsr_phases=dict(
                projection=projection_lease(),
                residual=compute(),
                local_rms=compute() + registers(2, "dest", "src0", "src1"),
                row_collective=row_reduce(),
            ),
        ),
        nodes=[
            dict(id=f"p{x}_{y}", tile=[x, y], place=[4 + x, 1 + y])
            for y in range(p)
            for x in range(p)
        ],
    )


def inputs(m, b):
    check(set(b) == {n["host"] for n in m["nodes"][:4]}, "composition input ports")
    arrays = []
    for n in m["nodes"][:4]:
        a = np.asarray(b[n["host"]], float).reshape(n["shape"])
        check(
            np.all(np.isfinite(a))
            and np.all(np.abs(a) <= effective_bound(n, m["input_bound"]))
            and np.array_equal(a, a.astype(np.float16).astype(float)),
            "finite exact-half bounded inputs",
        )
        arrays.append(a)
    return arrays


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "composition epochs")
    outputs = []
    for b in batches:
        x, w, r, g = inputs(m, b)
        z = (matmul(x, w) + r).astype(np.float16).astype(float)
        y = (
            z
            * g
            / np.sqrt(
                np.sum(z * z, axis=1)[:, None] / z.shape[1] + m["nodes"][6]["epsilon"]
            )
        )
        outputs.append(
            {
                m["nodes"][-1]["host"]: y.astype(np.float16)
                .astype(float)
                .ravel()
                .tolist()
            }
        )
    return outputs, {}


def reference(s, x, w, r, g):
    from projection_reference import project
    from mesh_common import pack_tiles

    projection, history, left, right, _ = project(s["projection_stage"], x, w)
    z = (projection + r).astype(np.float16).astype(float)
    local, total, inverse, result = rms_reference(s, z, g)
    return result, dict(
        history=history,
        sum=pack_tiles(z, s["P"], s["P"], "F"),
        left_first=left,
        right_first=right,
        local_square_sum=local,
        reduced_square_sum=total,
        inverse=inverse,
    )


def generate(s, dest):
    rt = Path(__file__).parent / "runtime"
    dest = Path(dest)
    for out, source in (
        ("pe.csl", "projection_residual_rms_pe.csl"),
        ("layout.csl", "projection_residual_rms_layout.csl"),
        ("rms_local.csl", "rms_local.csl"),
        ("inference_comm.csl", "inference_comm.csl"),
        ("inference_routes.csl", "inference_routes.csl"),
        ("WaferLLM-LICENSE.txt", "waferllm-LICENSE.txt"),
    ):
        (dest / out).write_bytes((rt / source).read_bytes())
    (dest / "SOURCE-NOTICE.txt").write_text(
        "Source-derived projection/residual/RMS; MeshInfra/WaferLLM fd1c2daae37cd68706c03fc8009887ecee9900f8, Apache-2.0. Original h1 projection and co-designed communication; explicit correct square descriptor and row-vector inverse scaling, immutable inputs and private last-use buffer reuse. Local RMS extracted as synchronous CSL library. Supplied attention activation and independent weights/residual/gamma, not a full Prefill/model or hardware qualification.\n"
    )
