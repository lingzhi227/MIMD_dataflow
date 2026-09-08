"""Transport and exact all-PE witnesses for resident normalized branch fusion."""

import numpy as np
from frontend import check
from half_region_runtime import read
import mesh_batched_rms_sdk as rms_sdk
from mesh_batched_fanout import plan, inputs, reference, branches
from compiler_parameters import encode


def parameters(s):
    v = dict(
        P=s["P"],
        B=s["B"],
        Nt=s["Nt"],
        Ft=s["Ft"],
        C=s["projections"],
        groups=s["groups"],
        epsilon_bits=int(np.float16(s["epsilon"]).view(np.uint16)),
        sampled=int(s["instrumentation"] == "sampled"),
    )
    return encode({k: "u16" if k == "epsilon_bits" else "i16" for k in v}, v)


def extents(s):
    d = rms_sdk.extents(s)
    d.update(
        weights=s["projections"] * s["Nt"] * s["Ft"],
        projections=s["padded_projection"],
        partial=s["padded_projection"] if s["instrumentation"] == "sampled" else 1,
        progress=8,
    )
    return d


def packed(s, m, b):
    x, w, *ws = inputs(m, b)
    weights = np.zeros((s["P"], s["P"], s["projections"] * s["Nt"] * s["Ft"]))
    for y in range(s["P"]):
        for col in range(s["P"]):
            weights[y, col] = np.concatenate(
                [
                    q[
                        y * s["Nt"] : (y + 1) * s["Nt"],
                        col * s["Ft"] : (col + 1) * s["Ft"],
                    ].ravel()
                    for q in ws
                ]
            )
    return dict(
        X=rms_sdk.shard(s, x),
        W=np.repeat(w.reshape(s["P"], 1, s["Nt"]), s["P"], axis=1),
        weights=weights,
    )


def decode(s, m, d):
    a = np.asarray(d["projections"], np.uint16).view(np.float16)[0]
    return {
        binding["output"]: a[
            :, binding["offset"] : binding["offset"] + binding["length"]
        ]
        .reshape(s["P"], s["B"], s["Ft"])
        .transpose(1, 0, 2)
        .astype(float)
        .ravel()
        .tolist()
        for binding in s["branch_bindings"]
    }


def run(root):
    from half_region_runtime import run as execute

    execute(root, parameters, extents, packed, decode)


def audit_cases(s, m, batches, r, *, require_complete=True):
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
        "batched fanout lifecycle",
    )
    bits = lambda a: np.asarray(a, np.float16).view(np.uint16)
    reports = []
    for e, (b, d, out) in enumerate(zip(batches, r["diagnostics"], r["cases"])):
        check(set(d) == set(extents(s)), "batched fanout diagnostic ports")
        arrays = {}
        for k, n in extents(s).items():
            a = np.asarray(d[k])
            check(
                a.shape == (s["P"], s["P"], n)
                and np.issubdtype(a.dtype, np.integer)
                and np.all((a >= 0) & (a < 65536)),
                "batched fanout raw " + k,
            )
            arrays[k] = a.astype(np.uint16)
        for k, v in packed(s, m, b).items():
            np.testing.assert_array_equal(arrays[k], bits(v), err_msg="immutable " + k)
        np.testing.assert_array_equal(
            arrays["progress"], np.tile([1] * 7 + [e + 1], (s["P"], s["P"], 1))
        )
        check(np.all((arrays["queues"] & 248) == 248), "batched fanout queues drained")
        x, w, *ws = inputs(m, b)
        local, sums, norm, partial, total, outputs = reference(s, x, w, ws)
        np.testing.assert_array_equal(
            arrays["sums"],
            np.broadcast_to(bits(sums), arrays["sums"].shape),
            err_msg="RMS sum/padding",
        )
        np.testing.assert_array_equal(
            arrays["result"],
            bits(rms_sdk.shard(s, norm)),
            err_msg="normalized replicas",
        )
        np.testing.assert_array_equal(
            arrays["projections"],
            np.broadcast_to(bits(total), arrays["projections"].shape),
            err_msg="all branch results and row replicas",
        )
        if s["instrumentation"] == "sampled":
            hist = np.concatenate((local, np.broadcast_to(sums, local.shape)), axis=1)
            np.testing.assert_array_equal(
                arrays["history"],
                bits(np.repeat(hist[:, None, :], s["P"], axis=1)),
                err_msg="RMS witnesses",
            )
            np.testing.assert_array_equal(
                arrays["partial"], bits(partial), err_msg="all local branch witnesses"
            )
        else:
            check(
                np.all(arrays["history"] == 0) and np.all(arrays["partial"] == 0),
                "inactive observations",
            )
        check(out == decode(s, m, d), "batched fanout logical decoding")
        nominal_norm = (
            x * w / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + s["epsilon"])
        )
        accuracy = []
        for q, actual, binding in zip(ws, outputs, s["branch_bindings"]):
            nominal = nominal_norm @ q
            error = actual - nominal
            l2 = float(np.linalg.norm(error) / max(np.linalg.norm(nominal), 1e-30))
            peak = float(np.max(np.abs(error)) / max(np.max(np.abs(nominal)), 1e-30))
            check(
                np.all(np.isfinite(actual)) and l2 <= 0.02 and peak <= 0.03,
                "batched fanout fixed branch accuracy " + binding["output"],
            )
            accuracy.append(
                dict(output=binding["output"], relative_l2=l2, peak_scaled_error=peak)
            )
        t = arrays["timing"].astype(np.int64)
        cycles = sum(
            (t[:, :, i + 3] - t[:, :, i]) * (1 << (16 * i)) for i in range(3)
        ) % (1 << 48)
        check(np.all((cycles > 0) & (cycles < 2**32)), "batched fanout bounded cycles")
        reports.append(
            dict(
                epoch=e,
                branches=accuracy,
                all_pe_witnesses_exact=True,
                max_local_cycles=int(cycles.max()),
                cycles_per_pe=cycles.tolist(),
            )
        )
    return dict(
        passed=True,
        profile=s["profile"],
        epochs=len(reports),
        expected_epochs=m["epochs"],
        completed_calls=len(reports),
        full_run_passed=r["success"] and len(reports) == m["epochs"],
        cases=reports,
        scope="All packed branches, unchanged Y routes, same-instance dynamic extents, fixed independent original-input math. Simulator cycles, not hardware throughput.",
    )


def audit(root):
    from integrity import verify_bundle, verify_codegen

    verify_bundle(root)
    verify_codegen(root)
    s, m, b, r = (
        read(root, n)
        for n in ("schedule.json", "semantic.json", "batches.json", "results.json")
    )
    check(s == plan(m), "batched fanout frozen schedule")
    return audit_cases(s, m, b, r)
