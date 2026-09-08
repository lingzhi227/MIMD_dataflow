from source_tree import source_path, logical_name
"""Supplied shared-cache attention, explicit batch-major CSL phase contracts."""

import math
from pathlib import Path
import numpy as np
from frontend import check
from cache_attention_ir import canonical
from input_contracts import (
    verify_declarations,
    effective_bound,
    validate_batch,
    half_dot_bound,
    half_product_bound,
)
from binary16 import quantize, matmul
from batched_policies import MATMUL_X, MATMUL_Y, POINT_X

PROFILE = "mesh_cache_attention.v1"
SOFTMAX = dict(
    partition="sequence",
    axis="y",
    layout="batch_major",
    reduce="max_sum",
    provider="sdk_axis",
    accumulation="f16",
    collective="f32",
    math="sdk_half",
    compute="dsr",
    fp="relaxed",
)


def verify(module, epochs, bound):
    verify_declarations(module)
    m = canonical(module)
    x, q, k, v, w, t, score, sm, ctx, delta, add, out = m["nodes"]
    check(
        not m["states"] and not any("place" in n for n in m["nodes"]),
        "cache attention owns one region",
    )
    check(
        all(n.get("dtype") == "f16" for n in m["nodes"][:-1]),
        "cache attention half tensors",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 2,
        "cache attention execution bounds",
    )
    check(
        all(
            isinstance(node.get("shape"), list)
            and len(node["shape"]) == 2
            and all(type(v) is int and v > 0 for v in node["shape"])
            for node in m["nodes"][:-1]
        ),
        "cache attention positive rank-two tensors",
    )
    b, n = x["shape"]
    seq = k["shape"][0]
    check(
        q["shape"] == [b, n]
        and k["shape"] == v["shape"] == [seq, n]
        and w["shape"] == [n, n],
        "cache attention supplied tensor shapes",
    )
    check(
        t["shape"] == [n, seq]
        and score["shape"] == sm["shape"] == [b, seq]
        and ctx["shape"] == delta["shape"] == add["shape"] == [b, n],
        "cache attention intermediate shapes",
    )
    check("dataflow" not in t, "cache transpose is a serialized input view")
    out.update(dtype="f16", shape=[b, n])
    for node in m["nodes"]:
        node["interval"] = None
    m.update(profile=PROFILE, epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1, *, derived_query_bound=None, use_probability_mass=False):
    x, q, k, v, w, t, score, sm, ctx, delta, add, out = m["nodes"]
    b, n = x["shape"]
    seq = k["shape"][0]
    p = score.get("dataflow", {}).get("rows")
    check(
        partitions == 1 and type(p) is int and p in (4, 8, 16),
        "cache attention square region",
    )
    for node, policy in [
        (score, dict(MATMUL_X, result="sequence_rows")),
        (sm, SOFTMAX),
        (ctx, MATMUL_Y),
        (delta, MATMUL_X),
        (add, dict(POINT_X, axis="y")),
    ]:
        check(
            node.get("dataflow") == dict(policy, rows=p, cols=p),
            "cache attention explicit policy " + node["op"],
        )
    check(
        all(type(a) is int for a in (b, n, seq))
        and 1 <= b <= 16
        and n in (16, 64, 256, 1024)
        and 4 <= seq <= 512
        and n % p == seq % p == 0
        and n // p % 2 == seq // p % 2 == 0,
        "cache attention bounded even shards",
    )
    check(
        math.isfinite(sm["scale"]) and sm["scale"] == 1 / math.sqrt(n),
        "cache attention exact power-of-two inverse sqrt scale",
    )
    nt, st = n // p, seq // p
    pad = (b + 1) // 2 * 2
    cap = max(b * st, b * nt, pad)
    for node, size in ((score, nt), (ctx, st), (delta, nt)):
        check(
            node.get("accumulation") == "block_f32"
            and type(node.get("block_size")) is int
            and 1 <= node["block_size"] <= size
            and size % node["block_size"] == 0,
            "cache contraction requires explicit half partial / float merge dividing its shard",
        )
    mode = m.get("instrumentation", "sampled")
    check(mode in ("sampled", "counters"), "cache attention observations")
    check(
        max(nt * st, nt * nt, b * st, b * nt, p * pad) <= 32767,
        "cache attention descriptor/index range",
    )
    xb, qb, kb, vb, wb = [effective_bound(a, m["input_bound"]) for a in (x, q, k, v, w)]
    check(
        max(xb, qb, kb, vb) <= 1,
        "initial cache attention declared activation/cache bound <=1",
    )
    # A parent verified graph can fill the query slot from its own range proof.
    # Standalone DSL verification never supplies this parameter and keeps its
    # original input domain. The virtual query is a zero-bound interface stub.
    if derived_query_bound is not None:
        check(
            qb == 0
            and type(derived_query_bound) in (int, float)
            and math.isfinite(derived_query_bound)
            and 0 <= derived_query_bound <= 65504,
            "derived cache query requires a finite parent certificate and zero-bound stub",
        )
        qb = derived_query_bound
    check(
        not use_probability_mass or derived_query_bound is not None,
        "probability-mass propagation belongs to a derived-query parent graph",
    )

    def reduce_bound(value):
        check(p * value <= 65504, "cache attention collective overflow")
        return quantize(p * value)

    from blocked_matmul import finite_bound as local_bound

    sl = local_bound(qb, kb, nt, score["block_size"])
    sr = reduce_bound(sl)
    scaled = half_product_bound(sr, sm["scale"])
    check(2 * scaled <= 65504, "cache attention stable difference range")
    # exp in [0,1], with one exact exp(0)=1 at the actual global maximum.
    # Monotone nonnegative half addition preserves >=1 and is bounded by St;
    # SDK f32 SUM then half narrow gives denominator in [1,S] (S<=512).
    # Rounded reciprocal <=1, hence each rounded probability <=1.
    cl = local_bound(1, vb, st, ctx["block_size"])
    cr = reduce_bound(cl)
    mass_certificate = None
    if use_probability_mass:
        from positive_normalization_bounds import weighted_contraction

        mass_certificate = weighted_contraction(seq, st, p, ctx["block_size"], vb)
        cl, cr = mass_certificate["local"], mass_certificate["reduced"]
    dl = local_bound(cr, wb, nt, delta["block_size"])
    dr = reduce_bound(dl)
    check(dr + xb <= 65504, "cache attention residual overflow")
    alloc = dict(
        X=2 * b * nt,
        Q=2 * b * nt,
        K=2 * nt * st,
        V=2 * nt * st,
        W=2 * nt * nt,
        score=2 * b * st,
        scaled=2 * b * st,
        exponents=2 * b * st,
        probability=2 * b * st,
        local_max=2 * pad,
        maximum=2 * pad,
        local_sum=2 * pad,
        sums=2 * pad,
        context=2 * b * nt,
        delta=2 * b * nt,
        result=2 * b * nt,
        score_partial=2 * b * st if mode == "sampled" else 2,
        context_partial=2 * b * nt if mode == "sampled" else 2,
        delta_partial=2 * b * nt if mode == "sampled" else 2,
        collective_send=4 * cap,
        collective_reduced=4 * cap,
        max_send=4 * pad,
        max_gathered=4 * p * pad,
    )
    blocks = [node["block_size"] for node in (score, ctx, delta)]
    for role, size, columns, block_size in zip(
        ("score", "value", "output"), (nt, st, nt), (st, nt, nt), blocks
    ):
        alloc[role + "_block_partial"] = 2 * (columns if block_size < size else 1)
        alloc[role + "_block_total"] = 4 * (columns if block_size < size else 1)
    memory = dict(alloc, protocol_descriptors=1024, code_stack_reserve=20480)
    check(sum(memory.values()) <= 49152, "cache attention PE memory")
    resources = dict(
        colors=[0, 1, 4, 5],
        input_queues=[2, 3, 4, 5],
        output_queues=[2, 3, 4, 5],
        local_tasks=[10, 11, 14, 15, 16, 17],
        microthreads=[],
        dsr_ownership="Local DSR1/2 only after SUM or MAX caller callback; SDK X DSR1/Y DSR2. Shared SDK instances, never concurrent operations.",
        completion="Provider clears busy before caller callback; this releases local buffers/DSRs, not a global barrier.",
    )
    from sdk2101_resources import check_default_memcpy

    resources["sdk_default_memcpy"] = check_default_memcpy(resources)
    s = dict(
        profile=PROFILE,
        P=p,
        rows=p,
        cols=p,
        B=b,
        N=n,
        S=seq,
        Nt=nt,
        St=st,
        score_block=blocks[0],
        value_block=blocks[1],
        output_block=blocks[2],
        padded_batches=pad,
        collective_capacity=cap,
        epochs=m["epochs"],
        scale=sm["scale"],
        instrumentation=mode,
        numeric_allocations=alloc,
        memory_per_pe=memory,
        resources=resources,
        numerical_bounds=dict(
            **(
                dict(probability_mass_certificate=mass_certificate, derived_query=qb)
                if mass_certificate is not None
                else {}
            ),
            score_local=sl,
            score=sr,
            scaled=scaled,
            difference=quantize(2 * scaled),
            exponent=[0, 1],
            denominator=[1, seq],
            probability=[0, 1],
            context_local=cl,
            context=cr,
            delta_local=dl,
            delta=dr,
            result=quantize(dr + xb),
            scope="Finite bounds under direct SDK math contracts; original-input stage accuracy must be gated separately",
        ),
        input_bindings={
            a["host"]: role
            for a, role in zip(m["nodes"][:5], ("X", "Q", "K", "V", "W"))
        },
        output_port=out["host"],
        stages=[
            "local score",
            "SDK X SUM",
            "scale/local maximum",
            "SDK Y MAX gather/broadcast",
            "exp/local sum",
            "SDK Y SUM",
            "normalize/local value",
            "SDK Y SUM",
            "local output projection",
            "SDK X SUM",
            "residual",
            "host readback",
        ],
        ownership=dict(
            query="feature X, replica Y",
            key="sequence Y, feature X; local feature-major view",
            value="sequence Y, feature X; local sequence-major",
            score="sequence Y, replica X",
            context="feature X, replica Y",
            weight="input feature X, output feature Y",
            output="feature Y, replica X",
            cache="shared read-only across queries; no append, mask or head selection",
            intermediate_host_transfer=False,
        ),
        nodes=[
            dict(id=f"p{xx}_{yy}", tile=[xx, yy], place=[4 + xx, 1 + yy])
            for yy in range(p)
            for xx in range(p)
        ],
    )
    from cache_attention_lifetimes import storage_plan

    s["storage_lifetimes"] = storage_plan(s)
    return s


def values(m, batch):
    validate_batch(m, batch)
    check(
        set(batch) == {n["host"] for n in m["nodes"][:5]}, "cache attention input names"
    )
    result = []
    for node in m["nodes"][:5]:
        a = np.asarray(batch[node["host"]], float).reshape(node["shape"])
        check(
            np.all(np.isfinite(a))
            and np.all(np.abs(a) <= effective_bound(node, m["input_bound"]))
            and np.array_equal(a, a.astype(np.float16).astype(float)),
            "exact bounded half cache attention input",
        )
        result.append(a)
    return result


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "cache attention epochs")
    outputs = []
    from blocked_matmul import evaluate as blocked

    qh = lambda a: np.asarray(a, np.float16).astype(float)
    for batch in batches:
        x, q, k, v, w = values(m, batch)
        score = blocked(q, k.T, m["nodes"][6]["block_size"]) * m["nodes"][7]["scale"]
        e = np.exp(score - score.max(axis=1, keepdims=True))
        prob = qh(e / e.sum(axis=1, keepdims=True))
        outputs.append(
            {
                m["nodes"][-1]["host"]: qh(
                    x
                    + blocked(
                        blocked(prob, v, m["nodes"][8]["block_size"]),
                        w,
                        m["nodes"][9]["block_size"],
                    )
                )
                .ravel()
                .tolist()
            }
        )
    return outputs, {}


def generate(s, dest):
    runtime = source_path("runtime")
    names = {
        "layout.csl": "cache_attention_layout.csl",
        "pe.csl": "cache_attention_pe.csl",
    }
    names.update(
        {
            n: n
            for n in [
                "batched_matmul_local.csl",
                "batched_matmul_blocked.csl",
                "batched_softmax_local.csl",
                "sdk_axis_reduce.csl",
                "sdk_axis_max.csl",
            ]
        }
    )
    for name, source in names.items():
        (Path(dest) / name).write_bytes((runtime / source).read_bytes())
