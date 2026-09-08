"""SDK transport and independent bit/protocol/standard-algorithm RESIDENT audit."""

import json, tempfile
from pathlib import Path
import numpy as np
from frontend import check
from mesh_normalized_matmul import plan, generate, inputs, reference
from mesh_common import pack_tiles, unpack_tiles
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
    )
    return encode({k: "u16" if k == "epsilon_bits" else "i16" for k in v}, v)


def run(root):
    import os, subprocess
    from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder
    from cerebras.sdk.sdk_utils import input_array_to_u32
    from mesh_common import sdk_runtime

    root = Path(root).resolve()
    os.chdir(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    rows, cols, mt, nt = s["P"], s["P"], s["Mt"], s["Nt"]
    cmd = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        f"--fabric-dims={cols+7},{rows+2}",
        "--fabric-offsets=4,1",
        parameters(s),
        "-o=out",
        "--memcpy",
        "--channels=1",
    ]
    (root / "sdk-command.json").write_text(json.dumps(cmd) + "\n")
    subprocess.run(cmd, check=True)
    runner = sdk_runtime(root)
    ids = {
        k: runner.get_id(k)
        for k in (
            "X",
            "W",
            "Q_weight",
            "result",
            "normalized",
            "history",
            "progress",
            "timing",
            "queues",
        )
    }
    runner.load()
    runner.run()
    runner.launch("init_task", nonblock=False)

    def put(name, a, n):
        runner.memcpy_h2d(
            ids[name],
            input_array_to_u32(np.asarray(a, np.float16).ravel(), 1, 1),
            0,
            0,
            cols,
            rows,
            n,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_16BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )

    def get(name, n):
        a = np.zeros(rows * cols * n, np.uint32)
        runner.memcpy_d2h(
            a,
            ids[name],
            0,
            0,
            cols,
            rows,
            n,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_16BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )
        return a.astype(np.uint16).reshape(rows, cols, n)

    r = dict(success=False, runtime_instances=1, cases=[], diagnostics=[], launches=[])
    try:
        for b in read(root, "batches.json"):
            x, w, q = inputs(m, b)
            put("X", pack_tiles(x, rows, cols, "F"), mt * nt)
            put("W", np.repeat(w.reshape(1, cols, nt), rows, axis=0), nt)
            from mesh_twohop import pack

            put("Q_weight", pack(x, q, cols)[1], nt * nt)
            initial_q = get("Q_weight", nt * nt).tolist()
            runner.launch("hls_main", nonblock=False)
            r["launches"].append("hls_main")
            d = {
                k: get(k, n).tolist()
                for k, n in [
                    ("X", mt * nt),
                    ("W", nt),
                    ("result", mt * nt),
                    (
                        "history",
                        cols * mt * nt if s["instrumentation"] == "sampled" else 1,
                    ),
                    ("normalized", mt * nt if s["instrumentation"] == "sampled" else 1),
                    ("progress", 4),
                    ("timing", 6),
                    ("queues", 2),
                ]
            }
            d["projection_input"] = initial_q
            values = unpack_tiles(
                np.asarray(d["result"], np.uint16).view(np.float16), mt, nt, "F"
            ).astype(float)
            r["cases"].append({m["nodes"][-1]["host"]: values.ravel().tolist()})
            r["diagnostics"].append(d)
            (root / "results.json").write_text(json.dumps(r) + "\n")
            print("RESIDENT HLS", len(r["cases"]), flush=True)
    finally:
        runner.stop()
    r["success"] = True
    (root / "results.json").write_text(json.dumps(r) + "\n")


def audit(root):
    from integrity import verify_bundle
    from mesh_twohop import pack

    root = Path(root)
    verify_bundle(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    batches = read(root, "batches.json")
    r = read(root, "results.json")
    check(s == plan(m), "resident schedule regeneration")
    with tempfile.TemporaryDirectory() as td:
        generate(s, td)
        for n in ("layout.csl", "pe.csl", "inference_comm.csl", "inference_routes.csl"):
            check(
                (root / n).read_bytes() == (Path(td) / n).read_bytes(),
                "resident source regeneration " + n,
            )
    check(
        r["success"]
        and r["runtime_instances"] == 1
        and len(r["cases"]) == len(r["diagnostics"]) == len(batches) == m["epochs"]
        and r["launches"] == ["hls_main"] * len(batches),
        "resident lifecycle",
    )
    p, mt, nt = s["P"], s["Mt"], s["Nt"]
    sample = s["instrumentation"] == "sampled"
    reports = []
    observations = 0
    bits = lambda a: np.asarray(a, np.float16).view(np.uint16)
    for epoch, (b, d, o) in enumerate(zip(batches, r["diagnostics"], r["cases"])):
        x, w, q = inputs(m, b)
        norm, history, target = reference(s, x, w, q)
        arrays = {}
        for key, n in [
            ("X", mt * nt),
            ("W", nt),
            ("projection_input", nt * nt),
            ("result", mt * nt),
            ("normalized", mt * nt if sample else 1),
            ("history", p * mt * nt if sample else 1),
            ("progress", 4),
            ("timing", 6),
            ("queues", 2),
        ]:
            v = np.asarray(d[key])
            check(
                v.shape == (p, p, n)
                and np.issubdtype(v.dtype, np.integer)
                and np.all((v >= 0) & (v < 65536)),
                "resident diagnostic " + key,
            )
            arrays[key] = v.astype(np.uint16)
        np.testing.assert_array_equal(arrays["X"], bits(pack_tiles(x, p, p, "F")))
        np.testing.assert_array_equal(
            arrays["W"], bits(np.repeat(w.reshape(1, p, nt), p, axis=0))
        )
        np.testing.assert_array_equal(
            arrays["projection_input"], bits(pack(x, q, p)[1])
        )
        np.testing.assert_array_equal(
            arrays["progress"], np.tile([1, p, 1, epoch + 1], (p, p, 1))
        )
        check(np.all((arrays["queues"] & 248) == 248), "resident owned queues drained")
        if sample:
            np.testing.assert_array_equal(
                arrays["normalized"], bits(pack_tiles(norm, p, p, "F"))
            )
            np.testing.assert_array_equal(
                arrays["history"],
                bits(history.transpose(0, 1, 2, 4, 3).reshape(p, p, -1)),
            )
            observations += p * p * (p + 1) * mt * nt
        else:
            check(
                np.all(arrays["normalized"] == 0) and np.all(arrays["history"] == 0),
                "resident counter observation inactive",
            )
        actual = unpack_tiles(arrays["result"].view(np.float16), mt, nt, "F").astype(
            float
        )
        np.testing.assert_array_equal(bits(actual), bits(target))
        check(set(o) == {m["nodes"][-1]["host"]}, "resident output port")
        np.testing.assert_array_equal(
            bits(o[m["nodes"][-1]["host"]]).ravel(), bits(actual).ravel()
        )
        check(np.all(np.isfinite(actual)), "resident finite output")
        nominal = (x * w / np.sqrt(np.mean(x * x, axis=1)[:, None] + s["epsilon"])) @ q
        err = actual - nominal
        scale = float(np.linalg.norm(nominal))
        relative = float(np.linalg.norm(err)) / max(scale, 1e-30)
        maximum = float(np.max(np.abs(err)))
        check(
            relative <= 0.015
            and maximum <= 0.02 * max(float(np.max(np.abs(nominal))), 1e-30),
            "resident standard normwise half accuracy",
        )
        t = arrays["timing"].astype(np.int64)
        cycles = sum(
            (t[:, :, i + 3] - t[:, :, i]) * (1 << (16 * i)) for i in range(3)
        ) % (1 << 48)
        check(np.all((cycles > 0) & (cycles < 2**32)), "resident timestamp interval")
        reports.append(
            dict(
                target_bits_exact=True,
                standard_relative_l2=relative,
                standard_max_abs_error=maximum,
                max_local_cycles=int(cycles.max()),
                cycles_per_pe=cycles.tolist(),
            )
        )
    report = dict(
        passed=True,
        profile=s["profile"],
        epochs=len(batches),
        actors=p * p,
        instrumentation=s["instrumentation"],
        internal_half_observations=observations,
        cases=reports,
        performance_scope="WSE3 simulator local intervals, not hardware throughput or end-to-end latency",
    )
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
