"""Validation-only native equations and target trajectory, separate from emission."""

import numpy as np
from frontend import check
from projected_cache_reference import (
    reference as prefix_reference,
    original_math as prefix_math,
    q,
)
from mesh_projected_cache import values
from mesh_projected_cache_sdk import packed as prefix_packed
from mesh_cache_attention_sdk import shard, unshard
from cache_attention_reference import matmul, accuracy
from sdk_axis_reference import half_sum
from sdk_math_reference import rms_inverse_f16, stable_silu_f16


from projected_cache_ffn_transport import inputs, packed


def original_math(s, batch):
    v, (wu, wg, wd) = inputs(s, batch)
    a = s["attention"]
    ns = s["graph"]["ffn"]["nodes"]
    stage = prefix_math(v, a["epsilon"], a["scale"])
    z = stage["result"]
    gamma = v[1]
    norm = z * gamma / np.sqrt(np.mean(z * z, axis=1, keepdims=True) + ns[5]["epsilon"])
    up = norm @ wu
    gate = norm @ wg
    e = np.exp(-np.abs(gate))
    activation = np.where(gate >= 0, gate / (1 + e), gate * e / (1 + e))
    hidden = up * activation
    delta = hidden @ wd
    stage.update(
        ffn_normalized=norm,
        up=up,
        gate=gate,
        activation=activation,
        hidden=hidden,
        ffn_delta=delta,
        final_result=z + delta,
    )
    return stage


def reference(s, batch):
    a = s["attention"]
    v, (wu, wg, wd) = inputs(s, batch)
    raw, stages = prefix_reference(a, v)
    p, b, nt, ft, pad = a["P"], a["B"], a["Nt"], s["Ft"], a["padded_batches"]
    ns = s["graph"]["ffn"]["nodes"]
    z = stages["result"]
    gamma = v[1]
    local = np.zeros((p, pad))
    for y in range(p):
        for j in range(nt):
            local[y, :b] = q(local[y, :b] + q(z[:, y * nt + j] ** 2))
    # Exact dyadic scaling is done in f32; no half conversion before the sum.
    send = np.asarray(local, np.float32) / np.float32(a["N"])
    total = send[-1].copy()
    for row in send[-2::-1]:
        total = np.asarray(total + row, np.float32)
    mean = q(total)
    inverse = np.asarray([rms_inverse_f16(m, 1, ns[5]["epsilon"]) for m in mean[:b]])
    norm = q(q(z * gamma) * inverse[:, None])
    partial = np.empty((p, p, 2 * b * ft))
    for y in range(p):
        for x in range(p):
            partial[y, x] = np.concatenate(
                [
                    matmul(
                        norm[:, y * nt : (y + 1) * nt],
                        w[y * nt : (y + 1) * nt, x * ft : (x + 1) * ft],
                        node["block_size"],
                    ).ravel()
                    for w, node in ((wu, ns[6]), (wg, ns[7]))
                ]
            )
    reduced = half_sum(partial, 0)
    projections = np.repeat(reduced[None], p, axis=0)
    up, gate = [
        unshard(a, projections[:, :, i * b * ft : (i + 1) * b * ft], "x", s["F"])
        for i in range(2)
    ]
    activation = np.vectorize(stable_silu_f16)(gate)
    hidden = q(up * activation)
    down_partial = np.empty((p, p, b * nt))
    for y in range(p):
        for x in range(p):
            down_partial[y, x] = matmul(
                hidden[:, x * ft : (x + 1) * ft],
                wd[x * ft : (x + 1) * ft, y * nt : (y + 1) * nt],
                ns[10]["block_size"],
            ).ravel()
    delta_parts = half_sum(down_partial, 1)
    delta_physical = np.repeat(delta_parts[:, None], p, axis=1)
    delta = unshard(a, delta_physical, "y", a["N"])
    result = q(z + delta)
    history = np.concatenate([local, np.repeat(mean[None], p, axis=0)], axis=1)
    raw.update(packed(s, batch))
    raw.update(
        ffn_normalized=shard(a, norm, "y"),
        ffn_square_scratch=shard(a, q(z * z), "y"),
        ffn_sums=np.tile(mean, (p, p, 1)),
        ffn_rms_history=np.repeat(history[:, None], p, axis=1),
        ffn_projections=projections,
        ffn_projection_partial=partial,
        ffn_activation=shard(a, activation, "x"),
        ffn_hidden=shard(a, hidden, "x"),
        ffn_delta=delta_physical,
        ffn_down_partial=down_partial,
        ffn_result=shard(a, result, "y"),
    )
    stages.update(
        ffn_normalized=norm,
        up=up,
        gate=gate,
        activation=activation,
        hidden=hidden,
        ffn_delta=delta,
        final_result=result,
    )
    return raw, stages


def native_graph(module, batch):
    """Typed tensor equation interpreter matching the executable C++ shell."""
    from blocked_matmul import evaluate as blocked
    from mesh_pair_rotation import reference as pair

    values = {}
    pending = list(module["nodes"])
    while pending:
        ready = [n for n in pending if all(i in values for i in n["inputs"])]
        check(ready, "native graph acyclic")
        for n in ready:
            op = n["op"]
            args = [values[i] for i in n["inputs"]]
            if op == "input":
                v = np.asarray(batch[n["host"]], float).reshape(n["shape"])
            elif op == "output":
                v = args[0]
            elif op == "rmsnorm":
                v = q(
                    args[0]
                    * args[1]
                    / np.sqrt(
                        np.mean(args[0] ** 2, axis=1, keepdims=True) + n["epsilon"]
                    )
                )
            elif op == "matmul":
                v = blocked(*args, n["block_size"])
            elif op == "rotate_pairs":
                v = q(pair(*args, "odd_even", False)[1])
            elif op == "transpose":
                v = args[0].T
            elif op == "softmax":
                e = np.exp(
                    args[0] * n["scale"]
                    - np.max(args[0] * n["scale"], axis=1, keepdims=True)
                )
                v = q(e / np.sum(e, axis=1, keepdims=True))
            elif op == "silu":
                x = args[0]
                e = np.exp(-np.abs(x))
                v = q(np.where(x >= 0, x / (1 + e), x * e / (1 + e)))
            elif op == "multiply":
                v = q(args[0] * args[1])
            elif op == "add":
                v = q(args[0] + args[1])
            else:
                raise ValueError("unsupported native tensor op " + op)
            values[n["id"]] = v
            pending.remove(n)
    return values


def actual_stages(s, raw):
    a = s["attention"]
    b, nt, ft = a["B"], a["Nt"], s["Ft"]
    half = lambda k: np.asarray(raw[k], np.uint16).view(np.float16).astype(float)
    result = {}
    for logical, physical, axis, width in (
        ("normalized", "normalized", "y", a["N"]),
        ("rotated_query", "Q", "x", a["N"]),
        ("rotated_key", "rotated_key", "x", a["N"]),
        ("score", "score", "y", a["S"]),
        ("probability", "probability", "y", a["S"]),
        ("context", "context", "x", a["N"]),
        ("delta", "delta", "y", a["N"]),
        ("result", "result", "y", a["N"]),
        ("ffn_normalized", "ffn_normalized", "y", a["N"]),
        ("activation", "ffn_activation", "x", s["F"]),
        ("hidden", "ffn_hidden", "x", s["F"]),
        ("ffn_delta", "ffn_delta", "y", a["N"]),
        ("final_result", "ffn_result", "y", a["N"]),
    ):
        result[logical] = unshard(a, half(physical), axis, width)
    for i, name in enumerate(("query", "key_projection", "value_projection")):
        result[name] = unshard(
            a, half("projections")[:, :, i * b * nt : (i + 1) * b * nt], "x", a["N"]
        )
    for i, name in enumerate(("up", "gate")):
        result[name] = unshard(
            a, half("ffn_projections")[:, :, i * b * ft : (i + 1) * b * ft], "x", s["F"]
        )
    return result


def audit_cases(s, m, batches, r, *, require_complete=True):
    from projected_cache_ffn_codegen import extents
    from mesh_projected_cache_ffn_sdk import decode

    p = s["attention"]["P"]
    epochs = s["attention"]["epochs"]
    count = len(r.get("diagnostics", []))
    check(
        type(r.get("success")) is bool
        and type(r.get("runtime_instances")) is int
        and r["runtime_instances"] == 1
        and 1 <= count <= epochs
        and len(batches) == epochs
        and (not r["success"] or count == epochs)
        and len(r.get("cases", [])) == count,
        "composed lifecycle and completed calls",
    )
    check(
        r.get("launches") == ["hls_main"] * count,
        "composed one outer launch per completed call",
    )
    check(
        not require_complete or (r["success"] and count == epochs),
        "complete composed batch required",
    )
    ports = extents(s)
    reports = []
    for e, (batch, raw, public) in enumerate(
        zip(batches, r["diagnostics"], r["cases"])
    ):
        check(set(raw) == set(ports), "composed exact physical port set")
        for name, length in ports.items():
            v = np.asarray(raw[name])
            check(
                v.shape == (p, p, length)
                and np.issubdtype(v.dtype, np.integer)
                and np.all((v >= 0) & (v < 65536)),
                "composed raw words " + name,
            )
        predicted, _ = reference(s, batch)
        for name, v in predicted.items():
            np.testing.assert_array_equal(
                np.asarray(raw[name], np.uint16),
                np.asarray(v, np.float16).view(np.uint16),
                err_msg=f"{e} composed trajectory {name}",
            )
        np.testing.assert_array_equal(raw["progress"], np.full((p, p, 1), e + 1))
        np.testing.assert_array_equal(raw["stages"], np.ones((p, p, 8), int))
        np.testing.assert_array_equal(
            raw["attention_progress"], np.tile([1] * 10 + [e + 1], (p, p, 1))
        )
        check(
            np.all((np.asarray(raw["queues"]) & 60) == 60),
            "composed SDK plane queues empty",
        )
        check(public == decode(s, m, raw), "composed public output binding")
        ideal = original_math(s, batch)
        observed = actual_stages(s, raw)
        metrics = {k: accuracy(v, ideal[k]) for k, v in observed.items()}
        mass = float(np.max(np.abs(observed["probability"].sum(axis=1) - 1)))
        check(mass <= 0.01, "composed probability mass accuracy")
        t = np.asarray(raw["timing"], np.uint64)
        ticks = lambda v: v[:, :, 0] + (v[:, :, 1] << 16) + (v[:, :, 2] << 32)
        cycles = (ticks(t[:, :, 3:]) - ticks(t[:, :, :3])) & np.uint64((1 << 48) - 1)
        check(
            np.all((cycles > 0) & (cycles < 1 << 40)),
            "composed actual timestamp intervals",
        )
        reports.append(
            dict(
                epoch=e,
                metrics=metrics,
                probability_mass_error=mass,
                min_cycles=int(cycles.min()),
                max_cycles=int(cycles.max()),
                numeric_halfwords=sum(v.size for v in predicted.values()),
            )
        )
    return dict(
        passed=True,
        completed_calls=count,
        complete=r["success"] and count == epochs,
        cases=reports,
        scope="Actual all-PE half words, immutable inputs, 18 original-input numerical stages, probability mass, lifecycle/queues and instrumented simulator cycles. No real-hardware throughput or whole-source performance claim.",
    )
