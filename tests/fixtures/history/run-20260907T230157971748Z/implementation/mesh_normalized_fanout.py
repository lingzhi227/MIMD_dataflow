"""Bounded normalized-projection fan-out; stage sharing follows explicit SSA edges.

Two/three independent square projections share one normalization and one input
alignment. The no-copy branch transition is validated by separate executed source
experiments; each generated HLS configuration still requires SDK qualification.
"""

import copy
from pathlib import Path
import numpy as np
from frontend import check
import mesh_normalized_matmul as single


def branches(m):
    for i in range(3, len(m["nodes"]), 3):
        b = copy.deepcopy(m)
        b["nodes"] = copy.deepcopy(m["nodes"][:3] + m["nodes"][i : i + 3])
        yield b


def verify(module, epochs, bound):
    from normalized_fanout_ir import canonical
    m = canonical(module)
    norm = m["nodes"][2]
    verified = list(branches(m))
    for b in verified:
        single.verify(b, epochs, bound)
    for n in m["nodes"]:
        n["interval"] = None
        if n["op"] == "output":
            n.update(shape=norm["shape"][:], dtype="f16")
    m.update(profile="mesh_normalized_fanout.v1", epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    bs = list(branches(m))
    plans = [single.plan(b, partitions) for b in bs]
    base = plans[0]
    count = len(bs)
    check(
        all(
            {k: v for k, v in p.items() if k != "nodes"}
            == {k: v for k, v in base.items() if k != "nodes"}
            for p in plans
        ),
        "normalized fan-out matching physical schedules",
    )
    check(base["N"] <= 256, "normalized fan-out first bounded feature size256")
    p, mt, nt = base["P"], base["Mt"], base["Nt"]
    mode = base["instrumentation"]
    memory = copy.deepcopy(base["memory_per_pe"])
    memory["additional_projection_arrays"] = 2 * (count - 1) * (nt * nt + mt * nt)
    memory["additional_prefix_observations"] = (
        2 * (count - 1) * p * mt * nt if mode == "sampled" else 0
    )
    memory["branch_ownership_observations"] = (
        2 * count * mt * nt if mode == "sampled" else 2
    )
    check(sum(memory.values()) <= 49152, "normalized fan-out PE memory budget")
    s = copy.deepcopy(base)
    s.update(
        profile="mesh_normalized_fanout.v1",
        projections=count,
        memory_per_pe=memory,
        branch_bindings=[
            dict(
                index=i,
                weight=b["nodes"][3]["host"],
                output=b["nodes"][5]["host"],
                node=b["nodes"][4]["id"],
            )
            for i, b in enumerate(bs)
        ],
    )
    s["stages"] = base["stages"][:2] + [
        dict(
            id=i + 2,
            operation="matmul",
            branch=i,
            rounds=p,
            completion="X/Y task join and final buffer swap; live aligned owner retained for next branch",
        )
        for i in range(count)
    ]
    s["buffers"].update(
        host_inputs="stable X/W and packed projection weight slab; freshly loaded each warm call",
        normalized="one shared semantic tensor, one destructive alignment; subsequent branches consume completed live buffer ownership, never hard-reset to a named slot",
        projection="independent weight/output slices; one shared weight communication scratch after branch join",
    )
    s["ownership_transition"] = dict(
        before_next_entry="exchange completed live/previous pointer roles; matmul entry swap restores live send owner",
        extra_copy=False,
        extra_fabric_alignment=False,
        source_qualification="requires separate executed pointer-ownership control",
    )
    return s


def inputs(m, b):
    ns = [m["nodes"][0], m["nodes"][1]] + [v["nodes"][3] for v in branches(m)]
    check(set(b) == {n["host"] for n in ns}, "normalized fan-out host input set")
    values = []
    for branch in branches(m):
        subset = {
            n["host"]: b[n["host"]]
            for n in [branch["nodes"][0], branch["nodes"][1], branch["nodes"][3]]
        }
        x, w, q = single.inputs(branch, subset)
        if not values:
            values = [x, w]
        values.append(q)
    return values


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "normalized fan-out epoch count")
    outputs = [{} for _ in batches]
    for branch in branches(m):
        subset = [
            {
                n["host"]: b[n["host"]]
                for n in [branch["nodes"][0], branch["nodes"][1], branch["nodes"][3]]
            }
            for b in batches
        ]
        out, _ = single.evaluate(branch, subset)
        for combined, one in zip(outputs, out):
            combined.update(one)
    return outputs, {}


def reference(s, x, w, weights):
    norms = []
    histories = []
    outputs = []
    for q in weights:
        norm, history, result = single.reference(s, x, w, q)
        norms.append(norm)
        histories.append(history)
        outputs.append(result)
    for norm in norms[1:]:
        np.testing.assert_array_equal(norm, norms[0])
    return norms[0], histories, outputs


def generate(s, dest):
    rt = Path(__file__).parent / "runtime"
    dest = Path(dest)
    for out, src in [
        ("pe.csl", "normalized_fanout_pe.csl"),
        ("layout.csl", "normalized_fanout_layout.csl"),
        ("inference_comm.csl", "inference_comm.csl"),
        ("inference_routes.csl", "inference_routes.csl"),
    ]:
        (dest / out).write_bytes((rt / src).read_bytes())
    (dest / "WaferLLM-LICENSE.txt").write_bytes(
        (rt / "waferllm-LICENSE.txt").read_bytes()
    )
    (dest / "SOURCE-NOTICE.txt").write_text(
        "Resident normalized projection fan-out, derived from MeshInfra/WaferLLM Prefill fd1c2daae37cd68706c03fc8009887ecee9900f8, Apache-2.0. Corrected per-row RMS scaling and host feature weights; completed double-buffer pointer ownership preserved across projections; one normalization/alignment and sequential shared communication resources; stable host slabs, branch/prefix/ownership observations. No full inference or unmodified-source claim.\n"
    )
