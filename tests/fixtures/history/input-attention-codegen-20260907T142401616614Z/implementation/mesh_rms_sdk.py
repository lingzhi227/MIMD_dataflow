"""SDK transport and independent bit/protocol/standard-algorithm RMS audit."""

import json, tempfile
from pathlib import Path
import numpy as np
from frontend import check
from mesh_rms import plan, generate, inputs, reference
from mesh_common import pack_tiles, unpack_tiles
from compiler_parameters import encode


def read(root, name):
    return json.loads((Path(root) / name).read_text())


def parameters(s):
    v = {k: s[k] for k in ("rows", "cols", "Mt", "Nt")}
    v.update(
        epsilon_bits=int(np.float16(s["epsilon"]).view(np.uint16)),
        sampled=int(s["instrumentation"] == "sampled"),
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
    rows, cols, mt, nt = s["rows"], s["cols"], s["Mt"], s["Nt"]
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
        for k in ("X", "W", "result", "history", "progress", "timing", "queues")
    }
    runner.load()
    runner.run()

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
            x, w = inputs(m, b)
            put("X", pack_tiles(x, rows, cols, "F"), mt * nt)
            put("W", np.repeat(w.reshape(1, cols, nt), rows, axis=0), nt)
            runner.launch("hls_main", nonblock=False)
            r["launches"].append("hls_main")
            d = {
                k: get(k, n).tolist()
                for k, n in [
                    ("X", mt * nt),
                    ("W", nt),
                    ("result", mt * nt),
                    ("history", 3 * mt if s["instrumentation"] == "sampled" else 1),
                    ("progress", 5),
                    ("timing", 6),
                    ("queues", 2),
                ]
            }
            values = unpack_tiles(
                np.asarray(d["result"], np.uint16).view(np.float16), mt, nt, "F"
            ).astype(float)
            r["cases"].append({m["nodes"][3]["host"]: values.ravel().tolist()})
            r["diagnostics"].append(d)
            (root / "results.json").write_text(json.dumps(r) + "\n")
            print("RMS HLS", len(r["cases"]), flush=True)
    finally:
        runner.stop()
    r["success"] = True
    (root / "results.json").write_text(json.dumps(r) + "\n")


def audit(root):
    from integrity import verify_bundle

    root = Path(root)
    verify_bundle(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    batches = read(root, "batches.json")
    r = read(root, "results.json")
    check(s == plan(m), "RMS schedule regeneration")
    with tempfile.TemporaryDirectory() as td:
        generate(s, td)
        for n in ("layout.csl", "pe.csl", "row_reduce.csl"):
            check(
                (root / n).read_bytes() == (Path(td) / n).read_bytes(),
                "RMS code regeneration " + n,
            )
    check(
        r["success"]
        and r["runtime_instances"] == 1
        and len(r["cases"]) == len(r["diagnostics"]) == len(batches) == m["epochs"]
        and r["launches"] == ["hls_main"] * len(batches),
        "RMS lifecycle",
    )
    rows, cols, mt, nt = s["rows"], s["cols"], s["Mt"], s["Nt"]
    reports = []
    observations = 0
    bits = lambda v: np.asarray(v, np.float16).view(np.uint16)
    for epoch, (b, d, output) in enumerate(zip(batches, r["diagnostics"], r["cases"])):
        x, w = inputs(m, b)
        local, total, inv, target = reference(s, x, w)
        a = {}
        for key, n in [
            ("X", mt * nt),
            ("W", nt),
            ("result", mt * nt),
            ("history", 3 * mt if s["instrumentation"] == "sampled" else 1),
            ("progress", 5),
            ("timing", 6),
            ("queues", 2),
        ]:
            v = np.asarray(d[key])
            check(
                v.shape == (rows, cols, n)
                and np.issubdtype(v.dtype, np.integer)
                and np.all((v >= 0) & (v < 65536)),
                "RMS diagnostic words " + key,
            )
            a[key] = v.astype(np.uint16)
        np.testing.assert_array_equal(a["X"], bits(pack_tiles(x, rows, cols, "F")))
        np.testing.assert_array_equal(
            a["W"], bits(np.repeat(w.reshape(1, cols, nt), rows, axis=0))
        )
        np.testing.assert_array_equal(
            a["progress"], np.tile([1, 1, 1, 1, epoch + 1], (rows, cols, 1))
        )
        check(np.all((a["queues"] & 88) == 88), "RMS drained owned queues")
        if s["instrumentation"] == "sampled":
            np.testing.assert_array_equal(
                a["history"].reshape(rows, cols, 3, mt),
                bits(np.stack([local, total, inv], axis=2)),
            )
            observations += rows * cols * 3 * mt
        else:
            check(np.all(a["history"] == 0), "RMS counter history inactive")
        actual = unpack_tiles(a["result"].view(np.float16), mt, nt, "F").astype(float)
        np.testing.assert_array_equal(bits(actual), bits(target))
        check(set(output) == {m["nodes"][3]["host"]}, "RMS logical output port")
        np.testing.assert_array_equal(
            bits(output[m["nodes"][3]["host"]]).ravel(), bits(actual).ravel()
        )
        nominal = (
            x * w / np.sqrt(np.sum(x * x, axis=1)[:, None] / s["N"] + s["epsilon"])
        )
        error = actual - nominal
        scale = float(np.linalg.norm(nominal))
        relative = (
            float(np.linalg.norm(error)) / scale
            if scale
            else float(np.linalg.norm(error))
        )
        # Fixed acceptance policy for bounded half reduction; no input-dependent tolerance tuning.
        check(
            relative <= 0.01
            and float(np.max(np.abs(error)))
            <= 0.015 * max(float(np.max(np.abs(nominal))), 1e-30),
            "RMS standard algorithm accuracy",
        )
        t = a["timing"].astype(np.int64)
        ticks = sum((t[:, :, i + 3] - t[:, :, i]) * (1 << (16 * i)) for i in range(3))
        check(np.all((ticks > 0) & (ticks < 2**32)), "RMS bounded local interval")
        reports.append(
            dict(
                target_bits_exact=True,
                standard_relative_l2=relative,
                standard_max_abs=float(np.max(np.abs(error))),
                max_local_cycles=int(ticks.max()),
                cycles_per_pe=ticks.tolist(),
            )
        )
    report = dict(
        passed=True,
        profile=s["profile"],
        epochs=len(batches),
        actors=rows * cols,
        instrumentation=s["instrumentation"],
        internal_half_observations=observations,
        cases=reports,
        scope="WSE3 SDK simulator; correct row RMSNorm with half source-order accumulation and SDK sqrt. No full Prefill/Decode or hardware throughput claim.",
        acceptance=dict(relative_l2=0.01, peak_scaled_max_abs=0.015),
    )
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
