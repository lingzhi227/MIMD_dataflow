"""Resident projection fan-out slab transport and branch ownership audit."""

import json, tempfile
from pathlib import Path
import numpy as np
from frontend import check
from mesh_normalized_fanout import plan, generate, inputs, reference
from mesh_common import pack_tiles, unpack_tiles
from mesh_twohop import pack, block_index
from compiler_parameters import encode


def read(root, name):
    return json.loads((Path(root) / name).read_text())


def parameters(s):
    p = s["P"]
    v = dict(
        P=p,
        dim_p_pe=s["Nt"],
        pes_p_head=p,
        pes_p_kv_head=p,
        head_dim_p_pe=s["Nt"],
        seq_len_p_pe=s["Mt"],
        ffn_dim_p_pe=s["Nt"],
        sampled=int(s["instrumentation"] == "sampled"),
        epsilon_bits=int(np.float16(s["epsilon"]).view(np.uint16)),
        projections=s["projections"],
    )
    return encode({k: "u16" if k == "epsilon_bits" else "i16" for k in v}, v)


def extents(s):
    p, mt, nt, b = s["P"], s["Mt"], s["Nt"], s["projections"]
    sample = s["instrumentation"] == "sampled"
    return dict(
        X=mt * nt,
        W=nt,
        result=b * mt * nt,
        normalized=mt * nt if sample else 1,
        history=b * p * mt * nt if sample else 1,
        reuse=b * mt * nt if sample else 1,
        progress=4,
        timing=6,
        queues=2,
    )


def weight_slab(s, x, weights):
    p = s["P"]
    return np.stack(
        [pack(x, q, p)[1].reshape(p, p, -1) for q in weights], axis=2
    ).reshape(p, p, -1)


def outputs(s, raw):
    p, mt, nt, b = s["P"], s["Mt"], s["Nt"], s["projections"]
    a = np.asarray(raw, np.uint16).view(np.float16).reshape(p, p, b, mt * nt)
    return {
        v["output"]: unpack_tiles(a[:, :, i, :], mt, nt, "F")
        .astype(float)
        .ravel()
        .tolist()
        for i, v in enumerate(s["branch_bindings"])
    }


def run(root):
    import os, subprocess
    from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder
    from cerebras.sdk.sdk_utils import input_array_to_u32
    from mesh_common import sdk_runtime

    root = Path(root).resolve()
    os.chdir(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    p, mt, nt = s["P"], s["Mt"], s["Nt"]
    cmd = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        f"--fabric-dims={p+7},{p+2}",
        "--fabric-offsets=4,1",
        parameters(s),
        "-o=out",
        "--memcpy",
        "--channels=1",
    ]
    (root / "sdk-command.json").write_text(json.dumps(cmd) + "\n")
    subprocess.run(cmd, check=True)
    runner = sdk_runtime(root)
    ids = {k: runner.get_id(k) for k in list(extents(s)) + ["Q_weight"]}
    runner.load()
    runner.run()
    runner.launch("init_task", nonblock=False)

    def put(name, a, n):
        runner.memcpy_h2d(
            ids[name],
            input_array_to_u32(np.asarray(a, np.float16).ravel(), 1, 1),
            0,
            0,
            p,
            p,
            n,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_16BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )

    def get(name, n):
        a = np.zeros(p * p * n, np.uint32)
        runner.memcpy_d2h(
            a,
            ids[name],
            0,
            0,
            p,
            p,
            n,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_16BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )
        return a.astype(np.uint16).reshape(p, p, n)

    def save_result(value):
        temporary = root / "results.json.tmp"
        temporary.write_text(json.dumps(value) + "\n")
        temporary.replace(root / "results.json")

    r = dict(success=False, runtime_instances=1, cases=[], diagnostics=[], launches=[])
    try:
        for batch in read(root, "batches.json"):
            x, w, *weights = inputs(m, batch)
            put("X", pack_tiles(x, p, p, "F"), mt * nt)
            put("W", np.repeat(w.reshape(1, p, nt), p, axis=0), nt)
            put("Q_weight", weight_slab(s, x, weights), s["projections"] * nt * nt)
            initial = get("Q_weight", s["projections"] * nt * nt).tolist()
            runner.launch("hls_main", nonblock=False)
            r["launches"].append("hls_main")
            d = {k: get(k, n).tolist() for k, n in extents(s).items()}
            d["projection_input"] = initial
            r["cases"].append(outputs(s, d["result"]))
            r["diagnostics"].append(d)
            save_result(r)
            checkpoint = audit_cases(
                s, m, read(root, "batches.json"), r, require_complete=False
            )
            (root / "completed-call-audit.json").write_text(
                json.dumps(checkpoint, indent=2) + "\n"
            )
            print("NORMALIZED FANOUT HLS", len(r["cases"]), flush=True)
    finally:
        runner.stop()
    r["success"] = True
    save_result(r)


def audit(root):
    from integrity import verify_bundle

    root = Path(root)
    verify_bundle(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    bs = read(root, "batches.json")
    r = read(root, "results.json")
    check(s == plan(m), "fan-out schedule regeneration")
    with tempfile.TemporaryDirectory() as td:
        generate(s, td)
        for name in (
            "layout.csl",
            "pe.csl",
            "inference_comm.csl",
            "inference_routes.csl",
            "WaferLLM-LICENSE.txt",
            "SOURCE-NOTICE.txt",
        ):
            check(
                (root / name).read_bytes() == (Path(td) / name).read_bytes(),
                "fan-out generated artifact " + name,
            )
    report = audit_cases(s, m, bs, r)
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def audit_cases(s, m, bs, r, require_complete=True):
    """Pure completed-call checks; partial use never qualifies the full bundle."""
    completed = len(r["cases"])
    check(
        type(r["success"]) is bool
        and r["runtime_instances"] == 1
        and len(bs) == m["epochs"]
        and len(r["diagnostics"]) == completed
        and r["launches"] == ["hls_main"] * completed
        and 1 <= completed <= m["epochs"]
        and (not r["success"] or completed == m["epochs"]),
        "fan-out completed prefix lifecycle",
    )
    if require_complete:
        check(
            r["success"] and completed == m["epochs"], "fan-out full lifecycle required"
        )
    p, mt, nt, b = s["P"], s["Mt"], s["Nt"], s["projections"]
    sample = s["instrumentation"] == "sampled"
    bits = lambda a: np.asarray(a, np.float16).view(np.uint16)
    reports = []
    words = 0
    for epoch, (batch, d, o) in enumerate(zip(bs, r["diagnostics"], r["cases"])):
        x, w, *weights = inputs(m, batch)
        norm, histories, targets = reference(s, x, w, weights)
        a = {}
        for key, n in dict(extents(s), projection_input=b * nt * nt).items():
            v = np.asarray(d[key])
            check(
                v.shape == (p, p, n)
                and np.issubdtype(v.dtype, np.integer)
                and np.all((v >= 0) & (v < 65536)),
                "fan-out raw words " + key,
            )
            a[key] = v.astype(np.uint16)
        np.testing.assert_array_equal(a["X"], bits(pack_tiles(x, p, p, "F")))
        np.testing.assert_array_equal(
            a["W"], bits(np.repeat(w.reshape(1, p, nt), p, axis=0))
        )
        np.testing.assert_array_equal(
            a["projection_input"], bits(weight_slab(s, x, weights))
        )
        np.testing.assert_array_equal(
            a["progress"], np.tile([1, b * p, b, epoch + 1], (p, p, 1))
        )
        check(np.all((a["queues"] & 248) == 248), "fan-out owned queues drained")
        if sample:
            np.testing.assert_array_equal(
                a["normalized"], bits(pack_tiles(norm, p, p, "F"))
            )
            history = (
                np.stack(histories, axis=2)
                .transpose(0, 1, 2, 3, 5, 4)
                .reshape(p, p, -1)
            )
            np.testing.assert_array_equal(a["history"], bits(history))
            aligned = np.zeros((p, p, mt * nt))
            for y in range(p):
                for col in range(p):
                    k = block_index(p, y, col, 0)
                    aligned[y, col] = norm[
                        y * mt : (y + 1) * mt, k * nt : (k + 1) * nt
                    ].ravel(order="F")
            np.testing.assert_array_equal(
                a["reuse"],
                bits(np.repeat(aligned[:, :, None, :], b, axis=2).reshape(p, p, -1)),
            )
            words += p * p * (1 + b * (p + 1)) * mt * nt
        else:
            check(
                all(np.all(a[k] == 0) for k in ("normalized", "history", "reuse")),
                "fan-out counter observations inactive",
            )
        actual = outputs(s, a["result"])
        check(set(o) == set(actual), "fan-out output ports")
        branches = []
        for i, v in enumerate(s["branch_bindings"]):
            value = np.asarray(actual[v["output"]]).reshape(s["M"], s["N"])
            np.testing.assert_array_equal(bits(value), bits(targets[i]))
            np.testing.assert_array_equal(
                bits(o[v["output"]]).ravel(), bits(value).ravel()
            )
            nominal = (
                x * w / np.sqrt(np.mean(x * x, axis=1)[:, None] + s["epsilon"])
            ) @ weights[i]
            err = value - nominal
            relative = float(np.linalg.norm(err)) / max(
                float(np.linalg.norm(nominal)), 1e-30
            )
            maximum = float(np.max(np.abs(err)))
            check(
                relative <= 0.015
                and maximum <= 0.02 * max(float(np.max(np.abs(nominal))), 1e-30),
                "fan-out standard half normwise accuracy",
            )
            branches.append(
                dict(
                    index=i,
                    output=v["output"],
                    target_bits_exact=True,
                    standard_relative_l2=relative,
                    standard_max_abs_error=maximum,
                )
            )
        t = a["timing"].astype(np.int64)
        cycles = sum(
            (t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)
        ) % (1 << 48)
        check(np.all((cycles > 0) & (cycles < 2**32)), "fan-out bounded timestamp")
        reports.append(
            dict(
                branches=branches,
                max_local_cycles=int(cycles.max()),
                cycles_per_pe=cycles.tolist(),
            )
        )
    report = dict(
        passed=True,
        profile=s["profile"],
        epochs=completed,
        requested_epochs=m["epochs"],
        complete=bool(r["success"] and completed == m["epochs"]),
        qualification_scope=(
            "complete_bundle_audit"
            if require_complete
            else "completed_call_prefix_diagnostic"
        ),
        actors=p * p,
        projections=b,
        instrumentation=s["instrumentation"],
        internal_half_observations=words,
        cases=reports,
        performance_scope="WSE3 simulator local interval, no full inference/hardware/global latency claim",
    )
    return report
