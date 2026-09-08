"""Validation-only SDK trajectory and independent original-input attention math."""

import numpy as np
from frontend import check
from blocked_matmul import evaluate as blocked_matmul
from binary16 import matmul as half_matmul
from sdk_axis_reference import half_sum
from sdk_math_reference import exp_f16_nonpositive


def matmul(a, b, block_size):
    """Match the CSL direct DSR fast path, including a final half negative zero."""
    if block_size == np.asarray(a).shape[1]:
        return half_matmul(a, b)
    return blocked_matmul(a, b, block_size)


def q(a):
    return np.asarray(a, np.float16).astype(float)


def reference(s, inputs):
    from mesh_cache_attention_sdk import shard, unshard

    x, query, key, value, w = inputs
    p, b, nt, st = s["P"], s["B"], s["Nt"], s["St"]
    pad = s["padded_batches"]
    parts = np.zeros((p, p, b * st))
    for y in range(p):
        for col in range(p):
            parts[y, col] = matmul(
                query[:, col * nt : (col + 1) * nt],
                key[y * st : (y + 1) * st, col * nt : (col + 1) * nt].T,
                s["score_block"],
            ).ravel()
    score = np.repeat(half_sum(parts, 1)[:, None], p, axis=1)
    scaled = q(score * s["scale"])
    lm = np.full((p, p, pad), -65504.0)
    lm[:, :, :b] = scaled.reshape(p, p, b, st).max(axis=3)
    maximum = np.repeat(lm.max(axis=0, keepdims=True), p, axis=0)
    shift = q(scaled.reshape(p, p, b, st) - maximum[:, :, :b, None])
    exp = np.asarray([exp_f16_nonpositive(a) for a in shift.ravel()]).reshape(
        p, p, b, st
    )
    ls = np.zeros((p, p, pad))
    for i in range(st):
        ls[:, :, :b] = q(ls[:, :, :b] + exp[:, :, :, i])
    sums = np.repeat(half_sum(ls, 0)[None], p, axis=0)
    prob = q(exp * q(1 / sums[:, :, :b, None]))
    prob = prob.reshape(p, p, b * st)
    cp = np.zeros((p, p, b * nt))
    for y in range(p):
        for col in range(p):
            cp[y, col] = matmul(
                prob[y, col].reshape(b, st),
                value[y * st : (y + 1) * st, col * nt : (col + 1) * nt],
                s["value_block"],
            ).ravel()
    context = np.repeat(half_sum(cp, 0)[None], p, axis=0)
    dp = np.zeros_like(cp)
    for y in range(p):
        for col in range(p):
            dp[y, col] = matmul(
                context[y, col].reshape(b, nt),
                w[col * nt : (col + 1) * nt, y * nt : (y + 1) * nt],
                s["output_block"],
            ).ravel()
    delta = np.repeat(half_sum(dp, 1)[:, None], p, axis=1)
    result = q(shard(s, x, "y") + delta)
    raw = dict(
        score_partial=parts,
        score=score,
        scaled=scaled,
        local_max=lm,
        maximum=maximum,
        exponents=exp.reshape(p, p, b * st),
        local_sum=ls,
        sums=sums,
        probability=prob,
        context_partial=cp,
        context=context,
        delta_partial=dp,
        delta=delta,
        result=result,
    )
    stages = {
        name: unshard(s, raw[name], axis, width)
        for name, axis, width in [
            ("score", "y", s["S"]),
            ("probability", "y", s["S"]),
            ("context", "x", s["N"]),
            ("delta", "y", s["N"]),
            ("result", "y", s["N"]),
        ]
    }
    return raw, stages


def original_math(inputs, scale):
    x, query, key, value, w = inputs
    score = query @ key.T
    scaled = score * scale
    e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    prob = e / e.sum(axis=1, keepdims=True)
    context = prob @ value
    delta = context @ w
    return dict(
        score=score, probability=prob, context=context, delta=delta, result=x + delta
    )


def accuracy(actual, expected):
    error = np.asarray(actual) - expected
    l2 = float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-12))
    peak = float(np.max(np.abs(error)) / max(np.max(np.abs(expected)), 1e-12))
    check(
        np.all(np.isfinite(actual)) and l2 <= 0.02 and peak <= 0.03,
        "cache attention original-input stage accuracy",
    )
    return dict(relative_l2=l2, relative_peak=peak)


def audit_cases(s, m, batches, r, *, require_complete=True):
    from mesh_cache_attention_sdk import values, packed, extents, decode

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
        and (not require_complete or (r["success"] and count == m["epochs"])),
        "cache attention lifecycle",
    )
    reports = []
    for epoch, (batch, d, out) in enumerate(zip(batches, r["diagnostics"], r["cases"])):
        check(set(d) == set(extents(s)), "cache attention diagnostic ports")
        for name, length in extents(s).items():
            a = np.asarray(d[name])
            check(
                a.shape == (p, p, length)
                and np.issubdtype(a.dtype, np.integer)
                and np.all((a >= 0) & (a < 65536)),
                "cache attention raw words " + name,
            )
        arrays = {k: np.asarray(v, np.uint16) for k, v in d.items()}
        inputs = values(m, batch)
        raw, stages = reference(s, inputs)
        raw.update(packed(s, m, batch))
        for name, expected in raw.items():
            if s["instrumentation"] == "counters" and name.endswith("_partial"):
                continue
            np.testing.assert_array_equal(
                arrays[name],
                np.asarray(expected, np.float16).view(np.uint16),
                err_msg=f"epoch{epoch} {name}",
            )
        np.testing.assert_array_equal(
            arrays["progress"], np.tile([1] * 7 + [epoch + 1], (p, p, 1))
        )
        check(
            np.all((arrays["queues"] & 60) == 60), "cache attention SDK queues drained"
        )
        check(out == decode(s, m, d), "cache attention decoded output")
        metrics = {
            k: accuracy(v, original_math(inputs, s["scale"])[k])
            for k, v in stages.items()
        }
        t = arrays["timing"].astype(np.uint64)
        start = t[:, :, 0] + (t[:, :, 1] << 16) + (t[:, :, 2] << 32)
        end = t[:, :, 3] + (t[:, :, 4] << 16) + (t[:, :, 5] << 32)
        cycles = (end - start) & ((1 << 48) - 1)
        check(
            np.all((cycles > 0) & (cycles < 2**32)), "cache attention positive timing"
        )
        reports.append(
            dict(epoch=epoch, stages=metrics, max_pe_cycles=int(cycles.max()))
        )
    return dict(
        passed=True,
        epochs=count,
        expected_epochs=m["epochs"],
        full_run_passed=r["success"] and count == m["epochs"],
        cases=reports,
        scope="All-PE raw target trajectory and original-input math; simulator cycles, not source/hardware performance qualification",
    )
