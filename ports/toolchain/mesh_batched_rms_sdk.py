"""Pure transport plus frozen bit/protocol audit for feature-sharded batched RMS."""

from pathlib import Path
import numpy as np
from frontend import check
from half_region_runtime import read
from mesh_batched_rms import plan, inputs, reference
from input_contracts import validate_batch
from compiler_parameters import encode


def parameters(s):
    values = dict(
        P=s["P"],
        B=s["B"],
        Nt=s["Nt"],
        groups=s["groups"],
        epsilon_bits=int(np.float16(s["epsilon"]).view(np.uint16)),
        sampled=int(s["instrumentation"] == "sampled"),
    )
    return encode({k: "u16" if k == "epsilon_bits" else "i16" for k in values}, values)


def extents(s):
    return dict(
        X=s["length"],
        W=s["Nt"],
        result=s["length"],
        sums=s["padded_batches"],
        history=2 * s["padded_batches"] if s["instrumentation"] == "sampled" else 1,
        progress=6,
        timing=6,
        queues=2,
    )


def shard(s, x):
    return np.repeat(
        np.asarray(x)
        .reshape(s["B"], s["P"], s["Nt"])
        .transpose(1, 0, 2)
        .reshape(s["P"], 1, s["length"]),
        s["P"],
        axis=1,
    )


def packed(s, m, b):
    validate_batch(m, b)
    x, w = inputs(m, b)
    return dict(
        X=shard(s, x), W=np.repeat(w.reshape(s["P"], 1, s["Nt"]), s["P"], axis=1)
    )


def decode(s, m, d):
    a = np.asarray(d["result"], np.uint16).view(np.float16)
    result = (
        a[:, 0, :]
        .reshape(s["P"], s["B"], s["Nt"])
        .transpose(1, 0, 2)
        .reshape(s["B"], s["N"])
    )
    return {m["nodes"][-1]["host"]: result.astype(float).ravel().tolist()}


def run(root):
    from half_region_runtime import run as execute

    execute(root, parameters, extents, packed, decode)


def audit_cases(s, m, batches, r):
    check(
        r.get("success")
        and r.get("runtime_instances") == 1
        and len(r.get("cases", []))
        == len(r.get("diagnostics", []))
        == len(batches)
        == m["epochs"]
        and r.get("launches") == ["hls_main"] * len(batches),
        "batched RMS lifecycle",
    )
    bits = lambda x: np.asarray(x, np.float16).view(np.uint16)
    cases = []
    for e, (b, d, out) in enumerate(zip(batches, r["diagnostics"], r["cases"])):
        validate_batch(m, b)
        check(set(d) == set(extents(s)), "batched RMS exact diagnostic ports")
        arrays = {}
        for name, n in extents(s).items():
            a = np.asarray(d[name])
            check(
                a.shape == (s["P"], s["P"], n)
                and np.issubdtype(a.dtype, np.integer)
                and np.all((a >= 0) & (a < 65536)),
                "batched RMS raw words " + name,
            )
            arrays[name] = a.astype(np.uint16)
        for name, value in packed(s, m, b).items():
            np.testing.assert_array_equal(
                arrays[name], bits(value), err_msg="immutable " + name
            )
        np.testing.assert_array_equal(
            arrays["progress"], np.tile([1, 1, 1, 1, 1, e + 1], (s["P"], s["P"], 1))
        )
        check(
            np.all((arrays["queues"] & 248) == 248), "batched RMS owned queues drained"
        )
        x, w = inputs(m, b)
        local, total, inverse, target = reference(s, x, w)
        expected_sum = np.broadcast_to(
            bits(total), (s["P"], s["P"], s["padded_batches"])
        )
        np.testing.assert_array_equal(
            arrays["sums"], expected_sum, err_msg="grouped sum and zero padding"
        )
        np.testing.assert_array_equal(
            arrays["result"], bits(shard(s, target)), err_msg="all result replicas"
        )
        if s["instrumentation"] == "sampled":
            hist = np.concatenate((local, np.broadcast_to(total, local.shape)), axis=1)
            expected = np.repeat(hist[:, None, :], s["P"], axis=1)
            np.testing.assert_array_equal(
                arrays["history"], bits(expected), err_msg="local/reduced witnesses"
            )
        else:
            check(np.all(arrays["history"] == 0), "inactive history")
        check(out == decode(s, m, d), "logical output matches raw result")
        nominal = x * w / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + s["epsilon"])
        error = target - nominal
        l2 = float(np.linalg.norm(error) / max(np.linalg.norm(nominal), 1e-30))
        peak = float(np.max(np.abs(error)) / max(np.max(np.abs(nominal)), 1e-30))
        check(
            np.all(np.isfinite(target)) and l2 <= 0.01 and peak <= 0.015,
            "batched RMS fixed mathematical accuracy",
        )
        t = arrays["timing"].astype(np.int64)
        cycles = sum(
            (t[:, :, i + 3] - t[:, :, i]) * (1 << (16 * i)) for i in range(3)
        ) % (1 << 48)
        check(
            np.all((cycles > 0) & (cycles < 2**32)),
            "batched RMS positive bounded cycles",
        )
        cases.append(
            dict(
                epoch=e,
                exact_replicas=True,
                padding_exact_zero=True,
                relative_l2=l2,
                peak_scaled_error=peak,
                max_local_cycles=int(cycles.max()),
                cycles_per_pe=cycles.tolist(),
            )
        )
    return dict(
        passed=True,
        profile=s["profile"],
        epochs=len(cases),
        cases=cases,
        scope="Actual half protocol and target-order bits, fixed RMS accuracy, all PE replicas. No in-call axis switch or hardware throughput qualification.",
    )


def audit(root):
    from integrity import verify_bundle, verify_codegen

    verify_bundle(root)
    verify_codegen(root)
    s, m, b, r = (
        read(root, n)
        for n in ("schedule.json", "semantic.json", "batches.json", "results.json")
    )
    check(s == plan(m), "batched RMS frozen plan regeneration")
    return audit_cases(s, m, b, r)
