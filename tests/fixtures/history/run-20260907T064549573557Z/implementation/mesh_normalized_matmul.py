"""Resident row normalization followed by contraction, with physical stage ownership.

Recognizes a typed two-operation subgraph, independent of user variable names.
The first bounded composition uses Prefill's co-designed communication resources;
it does not merge independently allocated standalone backends.
"""

import copy
from pathlib import Path
from inference_resources import projection_lease, row_reduce, compute, registers
import numpy as np
from frontend import check
from mesh_common import verify_matmul
from mesh_rms import verify as verify_rms, plan as plan_rms, reference as rms_reference
from mesh_twohop import block_index
from binary16 import matmul
from projection_contract import stage as projection_stage


def split(m):
    x, w, norm, q, mm, out = m["nodes"]
    normout = dict(
        id="__resident_norm_output",
        op="output",
        inputs=[norm["id"]],
        host="__normalized",
        shape=norm["shape"],
        dtype="f16",
    )
    r = copy.deepcopy(m)
    r["nodes"] = copy.deepcopy([x, w, norm, normout])
    lhs = dict(
        id=norm["id"],
        op="input",
        inputs=[],
        shape=norm["shape"],
        dtype="f16",
        host="__normalized",
    )
    g = copy.deepcopy(m)
    g["nodes"] = copy.deepcopy([lhs, q, mm, out])
    return r, g


def verify(module, epochs, bound):
    check(
        type(bound) is int and 0 <= bound <= 1,
        "resident first profile input magnitude bound0..1",
    )
    m = copy.deepcopy(module)
    check(
        len(m["nodes"]) == 6 and len({n["id"] for n in m["nodes"]}) == 6,
        "resident six unique SSA nodes",
    )
    by = {n["id"]: n for n in m["nodes"]}
    check(
        sorted(n["op"] for n in m["nodes"])
        == ["input", "input", "input", "matmul", "output", "rmsnorm"],
        "resident RMS/matmul typed subgraph",
    )
    norm = next(n for n in m["nodes"] if n["op"] == "rmsnorm")
    mm = next(n for n in m["nodes"] if n["op"] == "matmul")
    out = next(n for n in m["nodes"] if n["op"] == "output")
    check(
        len(norm["inputs"]) == 2
        and len(mm["inputs"]) == 2
        and all(v in by for n in m["nodes"] for v in n["inputs"]),
        "resident defined operand edges",
    )
    x, w = [by[v] for v in norm["inputs"]]
    q = by[mm["inputs"][1]]
    check(
        mm["inputs"][0] == norm["id"] and out["inputs"] == [mm["id"]],
        "resident producer/consumer dependencies",
    )
    check(
        all(n["op"] == "input" and n["inputs"] == [] for n in (x, w, q))
        and len({n["id"] for n in (x, w, q)}) == 3,
        "resident distinct source operands",
    )
    check(len({n["host"] for n in (x, w, q)}) == 3, "resident unique input ports")
    # Canonicalize only independent input declarations; source ids/lines remain.
    m["nodes"] = [x, w, norm, q, mm, out]
    r, g = split(m)
    r = verify_rms(r, epochs, bound)
    g = verify_matmul(g, epochs, bound, compute_modes=("dsr",))
    d = mm["dataflow"]
    check(
        d
        == dict(
            rows=norm["dataflow"]["rows"],
            cols=norm["dataflow"]["cols"],
            exchange="two_hop",
            initial_align="forward",
            reduce="local",
            overlap="double_buffer",
            fp="relaxed",
            compute="dsr",
        ),
        "resident source-compatible forward two-hop policy and shared mesh",
    )
    check(
        all(n.get("dtype") == "f16" for n in (x, w, norm, q, mm)),
        "resident binary16 tensors",
    )
    check(
        q["shape"] == [x["shape"][1], x["shape"][1]],
        "resident first contract requires square projection weights",
    )
    out.update(shape=mm["shape"][:], dtype="f16")
    for n in m["nodes"]:
        n["interval"] = None
    m.update(profile="mesh_normalized_matmul.v1", epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    r, g = split(m)
    projection = projection_stage(
        g["nodes"][0]["shape"], g["nodes"][1]["shape"], g["nodes"][2]["dataflow"]
    )
    rs = plan_rms(r, partitions)
    p = rs["rows"]
    mt, nt = rs["Mt"], rs["Nt"]
    check(p == rs["cols"] and p in (4, 8), "resident supported square mesh4/8")
    check(rs["M"] <= 256 and rs["N"] <= 512, "resident bounded matrix sizes")
    check(
        mt * nt % 4 == 0 and nt * nt % 4 == 0,
        "source communication requires full four-half packing groups",
    )
    mode = m.get("instrumentation", "sampled")
    history = (p + 1) * mt * nt if mode == "sampled" else 2
    memory = dict(
        resident_half_arrays=2 * (4 * mt * nt + 2 * nt * nt + nt + mt),
        observation_half_arrays=2 * history,
        witnesses_and_descriptor_reserve=2048,
        sdk_code_tasks_stack_reserve=16384,
    )
    check(sum(memory.values()) <= 49152, "resident PE memory budget")
    return dict(
        profile="mesh_normalized_matmul.v1",
        projection_stage=projection,
        M=rs["M"],
        N=rs["N"],
        P=p,
        Mt=mt,
        Nt=nt,
        epsilon=rs["epsilon"],
        epochs=m["epochs"],
        instrumentation=mode,
        memory_per_pe=memory,
        stages=[
            dict(
                id=0,
                operation="rmsnorm",
                completion="synchronous row reduce and row scale",
                output_ownership="logical column-major row/feature tiles",
            ),
            dict(
                id=1,
                operation="align",
                completion="all forward X sends and receives via left finish task",
                output_ownership="two-hop K block permutation; destructive private normalized buffer",
            ),
            dict(
                id=2,
                operation="matmul",
                completion="each round joins X and Y completions before buffer swap",
                rounds=p,
                output_ownership="logical column-major row/output-feature tiles",
            ),
        ],
        buffers=dict(
            host_inputs="stable X/W/Q_weight bindings; fresh loads each warm call",
            normalized="private tensor; single consumer; may be permuted after optional observation",
            scratch="RMS squares storage becomes X communication buffer after synchronous RMS completion",
            projection="private double-buffered weights; stable host weight binding reloaded each call",
        ),
        resources=dict(
            colors=list(range(1, 12)),
            input_queues=[3, 4, 5, 6, 7],
            output_queues=[3, 4, 5, 6, 7],
            microthreads=[0, 1, 2, 3],
            local_tasks=[19, 20, 25, 26],
            compute_dsrs=[1, 2],
            communication_dsrs=[2, 3, 4, 5, 6],
            explicit_dsr_phases=dict(
                local_normalization=compute() + registers(2, "dest", "src0", "src1"),
                row_collective=row_reduce(),
                projection=projection_lease(),
            ),
            reuse="RMS and matrix compute are sequential; matrix async traffic uses explicit UT0..3 and memory DSR3/4 and fabric DSR5/6",
        ),
        k_block_rule="cycle[(position(y)+position(x)-round)%P]",
        tile_orders=dict(X="F", weights="C", result="F"),
        nodes=[
            dict(id=f"p{x}_{y}", tile=[x, y], place=[4 + x, 1 + y])
            for y in range(p)
            for x in range(p)
        ],
    )


def inputs(m, b):
    nodes = [m["nodes"][i] for i in (0, 1, 3)]
    check(set(b) == {n["host"] for n in nodes}, "resident input ports")
    arrays = []
    for n in nodes:
        v = b[n["host"]]
        check(
            len(v) == np.prod(n["shape"]) and all(type(z) in (int, float) for z in v),
            "resident scalar extents/types",
        )
        a = np.asarray(v, float).reshape(n["shape"])
        check(
            np.all(np.isfinite(a)) and np.all(np.abs(a) <= m["input_bound"]),
            "resident finite input bounds",
        )
        check(
            np.array_equal(a, a.astype(np.float16).astype(float)),
            "resident representable half inputs",
        )
        arrays.append(a)
    return arrays


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "resident epochs")
    outputs = []
    for b in batches:
        x, w, q = inputs(m, b)
        norm = (
            (
                x
                * w
                / np.sqrt(
                    np.sum(x * x, axis=1)[:, None] / x.shape[1]
                    + m["nodes"][2]["epsilon"]
                )
            )
            .astype(np.float16)
            .astype(float)
        )
        outputs.append({m["nodes"][-1]["host"]: matmul(norm, q).ravel().tolist()})
    return outputs, {}


def reference(s, x, w, q):
    rs = dict(
        rows=s["P"],
        cols=s["P"],
        Mt=s["Mt"],
        Nt=s["Nt"],
        M=s["M"],
        N=s["N"],
        epsilon=s["epsilon"],
    )
    normalized = rms_reference(rs, x, w)[-1]
    p, mt, nt = s["P"], s["Mt"], s["Nt"]
    histories = np.zeros((p, p, p, mt, nt))
    result = np.zeros_like(x)
    for y in range(p):
        for col in range(p):
            tile = np.zeros((mt, nt))
            for step in range(p):
                block = block_index(p, y, col, step)
                for k in range(block * nt, (block + 1) * nt):
                    tile = np.asarray(
                        tile
                        + normalized[y * mt : (y + 1) * mt, k, None]
                        * q[None, k, col * nt : (col + 1) * nt],
                        np.float16,
                    ).astype(float)
                histories[y, col, step] = tile
            result[y * mt : (y + 1) * mt, col * nt : (col + 1) * nt] = tile
    check(np.all(np.isfinite(result)), "resident half projection overflow")
    return normalized, histories, result


def generate(s, dest):
    rt = Path(__file__).parent / "runtime"
    for out, src in [
        ("pe.csl", "normalized_matmul_pe.csl"),
        ("layout.csl", "normalized_matmul_layout.csl"),
        ("inference_comm.csl", "inference_comm.csl"),
        ("inference_routes.csl", "inference_routes.csl"),
    ]:
        (Path(dest) / out).write_text((rt / src).read_text())

    (Path(dest) / "WaferLLM-LICENSE.txt").write_bytes(
        (rt / "waferllm-LICENSE.txt").read_bytes()
    )
    (Path(dest) / "SOURCE-NOTICE.txt").write_text(
        "CSL resident normalization/projection runtime derived from WaferLLM Prefill\n"
        "https://github.com/MeshInfra/WaferLLM\n"
        "Pinned commit fd1c2daae37cd68706c03fc8009887ecee9900f8.\n"
        "Original license: Apache-2.0; see WaferLLM-LICENSE.txt.\n"
        "Modifications: isolated RMSNorm and projection, corrected per-row inverse scale,\n"
        "parameterized epsilon, stable HLS host bindings, phase/round diagnostics.\n"
        "Communication source retained; original complete Prefill is not claimed.\n"
    )
