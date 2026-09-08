"""Validation-only whole-graph half trajectory and original-input mathematics."""

import numpy as np
from frontend import check
from cache_attention_reference import matmul
from sdk_axis_reference import half_sum
from sdk_math_reference import rms_inverse_f16
from mesh_pair_rotation import reference as pair
from cache_attention_reference import reference as tail_reference, accuracy


def q(a):
    return np.asarray(a, np.float16).astype(float)


def reference(s, inputs):
    from mesh_cache_attention_sdk import shard, unshard

    x, gamma, wq, wk, wv, c, sn, key, value, wo = inputs
    p, b, nt = s["P"], s["B"], s["Nt"]
    pad = s["padded_batches"]
    local = np.zeros((p, pad))
    for y in range(p):
        for j in range(nt):
            local[y, :b] = q(local[y, :b] + q(x[:, y * nt + j] ** 2))
    sums = half_sum(local, 0)
    inverse = np.asarray([rms_inverse_f16(a, s["N"], s["epsilon"]) for a in sums[:b]])
    norm = q(q(x * gamma) * inverse[:, None])
    partial = np.zeros((p, p, 3 * b * nt))
    for y in range(p):
        for col in range(p):
            partial[y, col] = np.concatenate(
                [
                    matmul(
                        norm[:, y * nt : (y + 1) * nt],
                        w[y * nt : (y + 1) * nt, col * nt : (col + 1) * nt],
                        block,
                    ).ravel()
                    for w, block in zip((wq, wk, wv), s["projection_blocks"])
                ]
            )
    reduced = half_sum(partial, 0)
    projections = np.repeat(reduced[None], p, axis=0)
    branches = [
        unshard(s, projections[:, :, i * b * nt : (i + 1) * b * nt], "x", s["N"])
        for i in range(3)
    ]
    qp, rq = pair(branches[0], c, sn, "odd_even")
    kp, rk = pair(branches[1], c, sn, "odd_even")
    raw, stages = tail_reference(s, (x, rq, key, value, wo))

    def history(products):
        return np.stack(
            [shard(s, a, "x").reshape(p, p, b, nt // 2) for a in products], axis=3
        ).reshape(p, p, 2 * b * nt)

    qh, kh = history(qp), history(kp)
    rms_history = np.concatenate([local, np.repeat(sums[None], p, axis=0)], axis=1)
    raw.update(
        Q=shard(s, rq, "x"),
        normalized=shard(s, norm, "y"),
        rms_scratch=shard(s, q(x * x), "y"),
        rms_sums=np.tile(sums, (p, p, 1)),
        rms_history=np.repeat(rms_history[:, None], p, axis=1),
        projections=projections,
        projection_partial=partial,
        rotated_key=shard(s, rk, "x"),
        query_pair_history=qh,
        key_pair_history=kh,
        pair_scratch=kh.reshape(p, p, b, 2 * nt)[:, :, -1],
    )
    if s["instrumentation"] == "counters":
        for name in (
            "rms_history",
            "projection_partial",
            "query_pair_history",
            "key_pair_history",
            "score_partial",
            "context_partial",
            "delta_partial",
        ):
            raw[name] = np.zeros((p, p, 1))
    stages.update(
        normalized=norm,
        query=branches[0],
        key_projection=branches[1],
        value_projection=branches[2],
        rotated_query=rq,
        rotated_key=rk,
    )
    return raw, stages


def original_math(inputs, epsilon, scale):
    x, gamma, wq, wk, wv, c, sn, key, value, wo = inputs
    norm = x * gamma / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + epsilon)
    qq, kk, vv = [norm @ w for w in (wq, wk, wv)]
    rq, rk = [pair(a, c, sn, "odd_even", False)[1] for a in (qq, kk)]
    score = rq @ key.T
    e = np.exp(score * scale - (score * scale).max(axis=1, keepdims=True))
    prob = e / e.sum(axis=1, keepdims=True)
    context = prob @ value
    delta = context @ wo
    return dict(
        normalized=norm,
        query=qq,
        key_projection=kk,
        value_projection=vv,
        rotated_query=rq,
        rotated_key=rk,
        score=score,
        probability=prob,
        context=context,
        delta=delta,
        result=x + delta,
    )


def audit_cases(s, m, batches, r, *, require_complete=True):
    from mesh_projected_cache_sdk import values, packed, extents, decode

    count = len(r.get("cases", []))
    p = s["P"]
    check(
        type(r.get("success")) is bool
        and type(r.get("runtime_instances")) is int
        and r["runtime_instances"] == 1
        and 1 <= count <= m["epochs"]
        and len(r.get("diagnostics", [])) == count
        and len(batches) == m["epochs"]
        and r.get("launches") == ["hls_main"] * count
        and (not r["success"] or count == m["epochs"])
        and (not require_complete or r["success"] and count == m["epochs"]),
        "projected cache lifecycle",
    )
    reports = []
    for epoch, (batch, d, out) in enumerate(zip(batches, r["diagnostics"], r["cases"])):
        check(set(d) == set(extents(s)), "projected cache diagnostic ports")
        for name, length in extents(s).items():
            a = np.asarray(d[name])
            check(
                a.shape == (p, p, length)
                and np.issubdtype(a.dtype, np.integer)
                and np.all((a >= 0) & (a < 65536)),
                "projected cache raw words " + name,
            )
        arrays = {k: np.asarray(v, np.uint16) for k, v in d.items()}
        inputs = values(m, batch)
        raw, stages = reference(s, inputs)
        raw.update(packed(s, m, batch))
        check(
            set(raw) == set(arrays) - {"progress", "queues", "timing"},
            "projected cache every numeric port modeled",
        )
        for name, expected in raw.items():
            np.testing.assert_array_equal(
                arrays[name],
                np.asarray(expected, np.float16).view(np.uint16),
                err_msg=f"epoch{epoch} {name}",
            )
        np.testing.assert_array_equal(
            arrays["progress"], np.tile([1] * 10 + [epoch + 1], (p, p, 1))
        )
        check(
            np.all((arrays["queues"] & 60) == 60), "projected cache SDK queues drained"
        )
        check(out == decode(s, m, d), "projected cache three decoded outputs")
        expected = original_math(inputs, s["epsilon"], s["scale"])
        metrics = {k: accuracy(v, expected[k]) for k, v in stages.items()}
        mass = np.abs(np.sum(stages["probability"], axis=1) - 1)
        check(np.all(mass <= 0.01), "projected cache probability row mass")
        t = arrays["timing"].astype(np.uint64)
        start = t[:, :, 0] + (t[:, :, 1] << 16) + (t[:, :, 2] << 32)
        end = t[:, :, 3] + (t[:, :, 4] << 16) + (t[:, :, 5] << 32)
        cycles = (end - start) & ((1 << 48) - 1)
        check(np.all((cycles > 0) & (cycles < 2**32)), "projected cache timestamps")
        reports.append(
            dict(
                epoch=epoch,
                stages=metrics,
                max_probability_mass_error=float(mass.max()),
                max_pe_cycles=int(cycles.max()),
            )
        )
    return dict(
        passed=True,
        epochs=count,
        expected_epochs=m["epochs"],
        full_run_passed=r["success"] and count == m["epochs"],
        cases=reports,
        scope="All-PE whole-graph raw trajectory and eleven original-input mathematical stages; separate source comparison and linked memory required",
    )
