"""Generated stable softmax SDK transport, target-bit and standard-math audits."""

import json, tempfile
from pathlib import Path
import numpy as np
from frontend import check
from mesh_softmax import plan, generate, inputs, reference
from mesh_common import pack_tiles, unpack_tiles
from compiler_parameters import encode


def read(root, name):
    return json.loads((Path(root) / name).read_text())


def parameters(s):
    values = {k: s[k] for k in ("rows", "cols", "Mt", "Nt")}
    values.update(
        scale_bits=int(np.float16(s["scale"]).view(np.uint16)),
        sampled=int(s["instrumentation"] == "sampled"),
    )
    return encode({k: "u16" if k == "scale_bits" else "i16" for k in values}, values)


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
        for k in ("X", "result", "exponents", "history", "progress", "timing", "queues")
    }
    runner.load()
    runner.run()

    def get(name, n):
        v = np.zeros(rows * cols * n, np.uint32)
        runner.memcpy_d2h(
            v,
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
        return v.astype(np.uint16).reshape(rows, cols, n)

    result = dict(
        success=False, runtime_instances=1, cases=[], diagnostics=[], launches=[]
    )
    try:
        for b in read(root, "batches.json"):
            x = inputs(m, b)
            packed = pack_tiles(x, rows, cols, "F").astype(np.float16)
            runner.memcpy_h2d(
                ids["X"],
                input_array_to_u32(packed.ravel(), 1, 1),
                0,
                0,
                cols,
                rows,
                mt * nt,
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_16BIT,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False,
            )
            runner.launch("hls_main", nonblock=False)
            result["launches"].append("hls_main")
            d = {
                k: get(k, n).tolist()
                for k, n in [
                    ("X", mt * nt),
                    ("result", mt * nt),
                    ("history", 5 * mt if s["instrumentation"] == "sampled" else 1),
                    ("progress", 7),
                    ("timing", 6),
                    ("queues", 2),
                ]
            }
            d["exponents"] = (
                get("exponents", mt * nt).tolist()
                if s["instrumentation"] == "sampled"
                else None
            )
            result["cases"].append(
                {
                    m["nodes"][2]["host"]: unpack_tiles(
                        np.asarray(d["result"], np.uint16).view(np.float16), mt, nt, "F"
                    )
                    .astype(float)
                    .ravel()
                    .tolist()
                }
            )
            result["diagnostics"].append(d)
            (root / "results.json").write_text(json.dumps(result) + "\n")
            print("SOFTMAX HLS", len(result["cases"]), flush=True)
    finally:
        runner.stop()
    result["success"] = True
    (root / "results.json").write_text(json.dumps(result) + "\n")


def audit(root):
    from integrity import verify_bundle

    root = Path(root)
    verify_bundle(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    batches = read(root, "batches.json")
    r = read(root, "results.json")
    check(s == plan(m), "softmax schedule regeneration")
    with tempfile.TemporaryDirectory() as td:
        generate(s, td)
        for name in ("layout.csl", "pe.csl", "row_chain.csl"):
            check(
                (root / name).read_bytes() == (Path(td) / name).read_bytes(),
                "softmax generated CSL " + name,
            )
    check(
        r["success"]
        and r["runtime_instances"] == 1
        and len(r["cases"]) == len(r["diagnostics"]) == len(batches) == m["epochs"]
        and r["launches"] == ["hls_main"] * len(batches),
        "softmax lifecycle",
    )
    rows, cols, mt, nt = s["rows"], s["cols"], s["Mt"], s["Nt"]
    sampled = s["instrumentation"] == "sampled"
    reports = []
    observations = 0
    bits = lambda v: np.asarray(v, np.float16).view(np.uint16)
    for epoch, (b, d, logical) in enumerate(zip(batches, r["diagnostics"], r["cases"])):
        x = inputs(m, b)
        history, exponents, target = reference(s, x)
        a = {}
        shapes = [
            ("X", mt * nt),
            ("result", mt * nt),
            ("history", 5 * mt if sampled else 1),
            ("progress", 7),
            ("timing", 6),
            ("queues", 2),
        ]
        if sampled:
            shapes.append(("exponents", mt * nt))
        else:
            check(d["exponents"] is None, "softmax counters omit exponent observation")
        for key, n in shapes:
            v = np.asarray(d[key])
            check(
                v.shape == (rows, cols, n)
                and np.issubdtype(v.dtype, np.integer)
                and np.all((v >= 0) & (v < 65536)),
                "softmax diagnostic word " + key,
            )
            a[key] = v.astype(np.uint16)
        np.testing.assert_array_equal(a["X"], bits(pack_tiles(x, rows, cols, "F")))
        np.testing.assert_array_equal(
            a["progress"], np.tile([1] * 6 + [epoch + 1], (rows, cols, 1))
        )
        check(np.all((a["queues"] & 88) == 88), "softmax queues drained")
        if sampled:
            np.testing.assert_array_equal(
                a["history"].reshape(rows, cols, 5, mt), bits(history)
            )
            np.testing.assert_array_equal(
                a["exponents"], bits(pack_tiles(exponents, rows, cols, "F"))
            )
            observations += rows * cols * (5 * mt + mt * nt)
        else:
            check(np.all(a["history"] == 0), "softmax counter history inactive")
        actual = unpack_tiles(a["result"].view(np.float16), mt, nt, "F").astype(float)
        np.testing.assert_array_equal(bits(actual), bits(target))
        check(set(logical) == {m["nodes"][2]["host"]}, "softmax output port")
        np.testing.assert_array_equal(
            bits(logical[m["nodes"][2]["host"]]).ravel(), bits(actual).ravel()
        )
        v = x * s["scale"]
        ref = np.exp(v - v.max(axis=1)[:, None])
        ref /= ref.sum(axis=1)[:, None]
        error = actual - ref
        relative = float(np.linalg.norm(error) / np.linalg.norm(ref))
        maximum = float(np.max(np.abs(error)))
        row_error = float(np.max(np.abs(actual.sum(axis=1) - 1)))
        check(
            np.all(actual >= 0)
            and np.all(np.isfinite(actual))
            and relative <= 0.01
            and maximum <= 0.015 * float(ref.max())
            and row_error <= 0.01,
            "softmax standard accuracy/probability mass",
        )
        t = a["timing"].astype(np.int64)
        ticks = sum((t[:, :, i + 3] - t[:, :, i]) * (1 << (16 * i)) for i in range(3))
        check(np.all((ticks > 0) & (ticks < 2**32)), "softmax bounded timestamps")
        reports.append(
            dict(
                target_bits_exact=True,
                standard_relative_l2=relative,
                standard_max_abs=maximum,
                max_row_mass_error=row_error,
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
        acceptance=dict(relative_l2=0.01, peak_scaled_max_abs=0.015, row_mass=0.01),
        scope="Standard stable scaled row softmax with SDK half math; exact source-derived target bits. Original zero-max defect corrected. WSE3 simulator, no hardware/full-inference claim.",
    )
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
