"""SDK transport and independent bit/protocol/standard-algorithm GATED ACTIVATION audit."""

import json, tempfile
from pathlib import Path
import numpy as np
from frontend import check
from mesh_swiglu import plan, generate, inputs, reference
from mesh_common import pack_tiles, unpack_tiles
from compiler_parameters import encode


def read(root, name):
    return json.loads((Path(root) / name).read_text())


def parameters(s):
    v = {k: s[k] for k in ("rows", "cols", "length")}
    v["sampled"] = int(s["instrumentation"] == "sampled")
    return encode({k: "i16" for k in v}, v)


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
        for k in ("up", "gate", "result", "activated", "progress", "timing")
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
            up, gate = inputs(m, b)
            put("up", pack_tiles(up, rows, cols, "F"), mt * nt)
            put("gate", pack_tiles(gate, rows, cols, "F"), mt * nt)
            runner.launch("hls_main", nonblock=False)
            r["launches"].append("hls_main")
            d = {
                k: get(k, n).tolist()
                for k, n in [
                    ("up", mt * nt),
                    ("gate", mt * nt),
                    ("result", mt * nt),
                    ("activated", mt * nt if s["instrumentation"] == "sampled" else 1),
                    ("progress", 3),
                    ("timing", 6),
                ]
            }
            values = unpack_tiles(
                np.asarray(d["result"], np.uint16).view(np.float16), mt, nt, "F"
            ).astype(float)
            r["cases"].append({m["nodes"][-1]["host"]: values.ravel().tolist()})
            r["diagnostics"].append(d)
            (root / "results.json").write_text(json.dumps(r) + "\n")
            print("GATED ACTIVATION HLS", len(r["cases"]), flush=True)
    finally:
        runner.stop()
    r["success"] = True
    (root / "results.json").write_text(json.dumps(r) + "\n")


def audit(root):
    from integrity import verify_bundle
    from mesh_swiglu import accuracy

    root = Path(root)
    verify_bundle(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    batches = read(root, "batches.json")
    r = read(root, "results.json")
    check(s == plan(m), "gated activation schedule regeneration")
    with tempfile.TemporaryDirectory() as td:
        generate(s, td)
        for name in (
            "layout.csl",
            "pe.csl",
            "WaferLLM-LICENSE.txt",
            "SOURCE-NOTICE.txt",
        ):
            check(
                (root / name).read_bytes() == (Path(td) / name).read_bytes(),
                "gated activation artifact regeneration " + name,
            )
    check(
        r["success"]
        and r["runtime_instances"] == 1
        and len(r["cases"]) == len(r["diagnostics"]) == len(batches) == m["epochs"]
        and r["launches"] == ["hls_main"] * len(batches),
        "gated activation lifecycle",
    )
    rows, cols, mt, nt = s["rows"], s["cols"], s["Mt"], s["Nt"]
    sample = s["instrumentation"] == "sampled"
    reports = []
    words = 0
    bits = lambda a: np.asarray(a, np.float16).view(np.uint16)
    for epoch, (b, d, o) in enumerate(zip(batches, r["diagnostics"], r["cases"])):
        up, gate = inputs(m, b)
        activation, target = reference(up, gate)
        a = {}
        for key, n in [
            ("up", mt * nt),
            ("gate", mt * nt),
            ("result", mt * nt),
            ("activated", mt * nt if sample else 1),
            ("progress", 3),
            ("timing", 6),
        ]:
            v = np.asarray(d[key])
            check(
                v.shape == (rows, cols, n)
                and np.issubdtype(v.dtype, np.integer)
                and np.all((v >= 0) & (v < 65536)),
                "gated activation raw words " + key,
            )
            a[key] = v.astype(np.uint16)
        for key, v in [("up", up), ("gate", gate)]:
            np.testing.assert_array_equal(a[key], bits(pack_tiles(v, rows, cols, "F")))
        np.testing.assert_array_equal(
            a["progress"], np.tile([1, 1, epoch + 1], (rows, cols, 1))
        )
        if sample:
            np.testing.assert_array_equal(
                a["activated"], bits(pack_tiles(activation, rows, cols, "F"))
            )
            words += rows * cols * mt * nt
        else:
            check(
                np.all(a["activated"] == 0),
                "gated activation counter observation inactive",
            )
        actual = unpack_tiles(a["result"].view(np.float16), mt, nt, "F").astype(float)
        np.testing.assert_array_equal(bits(actual), bits(target))
        check(set(o) == {m["nodes"][-1]["host"]}, "gated activation output port")
        np.testing.assert_array_equal(
            bits(o[m["nodes"][-1]["host"]]).ravel(), bits(actual).ravel()
        )
        numerical = accuracy(up, gate, actual)
        t = a["timing"].astype(np.int64)
        cycles = sum(
            (t[:, :, i + 3] - t[:, :, i]) * (1 << (16 * i)) for i in range(3)
        ) % (1 << 48)
        check(
            np.all((cycles > 0) & (cycles < 2**32)),
            "gated activation bounded timestamp",
        )
        reports.append(
            dict(
                target_bits_exact=True,
                numerical=numerical,
                max_local_cycles=int(cycles.max()),
                cycles_per_pe=cycles.tolist(),
            )
        )
    report = dict(
        passed=True,
        profile=s["profile"],
        epochs=len(batches),
        actors=rows * cols,
        instrumentation=s["instrumentation"],
        internal_half_observations=words,
        cases=reports,
        performance_scope="WSE3 simulator local PE interval; no global latency or hardware throughput claim",
    )
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
