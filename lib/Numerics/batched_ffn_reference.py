"""Validation-only target trajectory and original-input standard FFN mathematics."""

import numpy as np
from frontend import check
from binary16 import matmul
from sdk_math_reference import rms_inverse_f16, stable_silu_f16


def q(a):
    return np.asarray(a, np.float16).astype(float)


def axis_sum(a, axis):
    # SDK reduce_fadds root0 linear chain: far endpoint toward root. This
    # association is checked against actual raw intermediates, never codegen.
    v = np.moveaxis(a, axis, 0)
    total = np.asarray(v[-1], np.float32)
    for x in v[-2::-1]:
        total = np.asarray(total + np.asarray(x, np.float32), np.float32)
    return q(total)


def reference(s, values):
    from mesh_batched_feed_forward_sdk import shard_y

    x, gamma, wu, wg, wd = values
    p, b, nt, ft = s["P"], s["B"], s["Nt"], s["Ft"]
    local = np.zeros((p, s["padded_batches"]))
    for y in range(p):
        for j in range(nt):
            local[y, :b] = q(local[y, :b] + q(x[:, y * nt + j] ** 2))
    total = axis_sum(local, 0)
    inv = np.array([rms_inverse_f16(v, s["N"], s["epsilon"]) for v in total[:b]])
    norm = q(q(x * gamma) * inv[:, None])
    part = np.zeros((p, p, 2 * b * ft))
    for y in range(p):
        ys = slice(y * nt, (y + 1) * nt)
        for col in range(p):
            xs = slice(col * ft, (col + 1) * ft)
            part[y, col] = np.concatenate(
                [matmul(norm[:, ys], w[ys, xs]).ravel() for w in (wu, wg)]
            )
    projected = axis_sum(part, 0)
    up = projected[:, : b * ft].reshape(p, b, ft).transpose(1, 0, 2).reshape(b, s["F"])
    gate = (
        projected[:, b * ft :].reshape(p, b, ft).transpose(1, 0, 2).reshape(b, s["F"])
    )
    act = np.array([stable_silu_f16(v) for v in gate.ravel()]).reshape(gate.shape)
    hidden = q(up * act)
    dp = np.zeros((p, p, b * nt))
    for y in range(p):
        ys = slice(y * nt, (y + 1) * nt)
        for col in range(p):
            xs = slice(col * ft, (col + 1) * ft)
            dp[y, col] = matmul(hidden[:, xs], wd[xs, ys]).ravel()
    reduced = axis_sum(dp, 1)
    delta = reduced.reshape(p, b, nt).transpose(1, 0, 2).reshape(b, s["N"])

    def shard_x(v):
        return np.repeat(
            v.reshape(b, p, ft).transpose(1, 0, 2).reshape(1, p, b * ft), p, axis=0
        )

    history = np.concatenate([local, np.repeat(total[None], p, axis=0)], axis=1)
    raw = dict(
        normalized=shard_y(s, norm),
        scratch=shard_y(s, q(x * x)),
        sums=np.tile(total, (p, p, 1)),
        history=np.repeat(history[:, None], p, axis=1),
        partial=part,
        projections=np.repeat(projected[None], p, axis=0),
        activation=shard_x(act),
        hidden=shard_x(hidden),
        down_partial=dp,
        delta=shard_y(s, delta),
        result=shard_y(s, q(x + delta)),
    )
    stages = dict(
        normalized=norm,
        up=up,
        gate=gate,
        activation=act,
        hidden=hidden,
        delta=delta,
        result=q(x + delta),
    )
    return raw, stages


def original_math(values, epsilon):
    x, gamma, wu, wg, wd = values
    norm = x * gamma / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + epsilon)
    up, gate = norm @ wu, norm @ wg
    e = np.exp(-np.abs(gate))
    activation = np.where(gate >= 0, gate / (1 + e), gate * e / (1 + e))
    hidden = up * activation
    delta = hidden @ wd
    return dict(
        normalized=norm,
        up=up,
        gate=gate,
        activation=activation,
        hidden=hidden,
        delta=delta,
        result=x + delta,
    )


def accuracy(actual, expected):
    error = np.asarray(actual) - expected
    l2 = float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-12))
    peak = float(np.max(np.abs(error)) / max(np.max(np.abs(expected)), 1e-12))
    check(
        np.all(np.isfinite(actual)) and l2 <= 0.02 and peak <= 0.03,
        "batch FFN original-input stage accuracy",
    )
    return dict(relative_l2=l2, relative_peak=peak)


def audit_cases(s, m, batches, r, *, require_complete=True):
    from mesh_batched_feed_forward_sdk import values, packed, extents, decode

    count = len(r.get("cases", []))
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
        "batch FFN lifecycle",
    )
    reports = []
    p = s["P"]
    for epoch, (batch, d, out) in enumerate(zip(batches, r["diagnostics"], r["cases"])):
        check(set(d) == set(extents(s)), "batch FFN diagnostic ports")
        for name, length in extents(s).items():
            a = np.asarray(d[name])
            check(
                a.shape == (p, p, length)
                and np.issubdtype(a.dtype, np.integer)
                and np.all((a >= 0) & (a < 65536)),
                "batch FFN raw words " + name,
            )
        arrays = {k: np.asarray(v, np.uint16) for k, v in d.items()}
        inputs = values(m, batch)
        raw, stages = reference(s, inputs)
        raw.update(packed(s, m, batch))
        for name, expected in raw.items():
            if s["instrumentation"] == "counters" and name in (
                "history",
                "partial",
                "down_partial",
            ):
                continue
            np.testing.assert_array_equal(
                arrays[name],
                np.asarray(expected, np.float16).view(np.uint16),
                err_msg=name,
            )
        np.testing.assert_array_equal(
            arrays["progress"], np.tile([1] * 7 + [epoch + 1], (p, p, 1))
        )
        check(np.all((arrays["queues"] & 60) == 60), "batch FFN SDK queues drained")
        check(out == decode(s, m, d), "batch FFN decoded output")
        independent = original_math(inputs, s["epsilon"])
        metrics = {k: accuracy(v, independent[k]) for k, v in stages.items()}
        t = arrays["timing"].astype(np.uint64)
        start = t[:, :, 0] + (t[:, :, 1] << 16) + (t[:, :, 2] << 32)
        end = t[:, :, 3] + (t[:, :, 4] << 16) + (t[:, :, 5] << 32)
        cycles = (end - start) & ((1 << 48) - 1)
        check(np.all((cycles > 0) & (cycles < 2**32)), "batch FFN positive timing")
        reports.append(
            dict(epoch=epoch, stages=metrics, max_pe_cycles=int(cycles.max()))
        )
    return dict(
        passed=True,
        epochs=len(reports),
        expected_epochs=m["epochs"],
        full_run_passed=r["success"] and count == m["epochs"],
        cases=reports,
        scope="Saved-call raw all-PE target trajectory and independent standard mathematics; simulator cycles only, no hardware or matched-source performance qualification",
    )
