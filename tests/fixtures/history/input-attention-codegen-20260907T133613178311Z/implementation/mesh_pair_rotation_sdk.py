"""SDK pair-transform transport, raw intermediate observations and target audit."""

import json, tempfile
from pathlib import Path
import numpy as np
from frontend import check
from mesh_pair_rotation import plan, generate, inputs, reference, accuracy
from mesh_common import pack_tiles, unpack_tiles
from compiler_parameters import encode


def read(root, name):
    return json.loads((Path(root) / name).read_text())


def parameters(s):
    v = {k: s[k] for k in ("rows", "cols", "Mt", "Nt")}
    v.update(
        broadcast=int(s["broadcast_coefficients"]),
        swapped=int(s["pair_order"] == "odd_even"),
        sampled=int(s["instrumentation"] == "sampled"),
    )
    return encode({k: "i16" for k in v}, v)


def packed(s, arrays):
    x, c, sn = arrays
    rows, cols = s["rows"], s["cols"]
    pack = lambda a: pack_tiles(a, rows, cols, "F")
    return dict(
        x=pack(x),
        cosine=(
            np.tile(pack_tiles(c, 1, cols, "F"), (rows, 1, 1))
            if s["broadcast_coefficients"]
            else pack(c)
        ),
        sine=(
            np.tile(pack_tiles(sn, 1, cols, "F"), (rows, 1, 1))
            if s["broadcast_coefficients"]
            else pack(sn)
        ),
    )


def extents(s):
    return dict(
        x=s["length"],
        cosine=s["coefficient_length"],
        sine=s["coefficient_length"],
        result=s["length"],
        history=2 * s["length"] if s["instrumentation"] == "sampled" else 1,
        progress=3,
        timing=6,
    )


def run(root):
    import os, subprocess
    from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder
    from cerebras.sdk.sdk_utils import input_array_to_u32
    from mesh_common import sdk_runtime

    root = Path(root).resolve()
    os.chdir(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    rows, cols = s["rows"], s["cols"]
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
    ids = {k: runner.get_id(k) for k in extents(s)}
    runner.load()
    runner.run()
    r = dict(success=False, runtime_instances=1, cases=[], diagnostics=[], launches=[])
    try:
        for b in read(root, "batches.json"):
            for name, a in packed(s, inputs(m, b)).items():
                runner.memcpy_h2d(
                    ids[name],
                    input_array_to_u32(np.asarray(a, np.float16).ravel(), 1, 1),
                    0,
                    0,
                    cols,
                    rows,
                    a.shape[-1],
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_16BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
            runner.launch("hls_main", nonblock=False)
            r["launches"].append("hls_main")
            d = {}
            for name, n in extents(s).items():
                raw = np.zeros(rows * cols * n, np.uint32)
                runner.memcpy_d2h(
                    raw,
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
                d[name] = raw.astype(np.uint16).reshape(rows, cols, n).tolist()
            value = unpack_tiles(
                np.asarray(d["result"], np.uint16).view(np.float16),
                s["Mt"],
                s["Nt"],
                "F",
            ).astype(float)
            r["cases"].append({m["nodes"][-1]["host"]: value.ravel().tolist()})
            r["diagnostics"].append(d)
            (root / "results.json").write_text(json.dumps(r) + "\n")
            print("PAIR ROTATION HLS", len(r["cases"]), flush=True)
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
    bs = read(root, "batches.json")
    r = read(root, "results.json")
    check(s == plan(m), "pair rotation schedule regeneration")
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
                "pair rotation regeneration " + name,
            )
    check(
        r["success"]
        and r["runtime_instances"] == 1
        and len(r["cases"]) == len(r["diagnostics"]) == len(bs) == m["epochs"]
        and r["launches"] == ["hls_main"] * len(bs),
        "pair rotation lifecycle",
    )
    rows, cols, mt, nt = s["rows"], s["cols"], s["Mt"], s["Nt"]
    bits = lambda a: np.asarray(a, np.float16).view(np.uint16)
    reports = []
    words = 0
    for epoch, (b, d, out) in enumerate(zip(bs, r["diagnostics"], r["cases"])):
        arrays = inputs(m, b)
        products, target = reference(*arrays, s["pair_order"])
        raw = {}
        for key, n in extents(s).items():
            a = np.asarray(d[key])
            check(
                a.shape == (rows, cols, n)
                and np.issubdtype(a.dtype, np.integer)
                and np.all((a >= 0) & (a < 65536)),
                "pair rotation raw words " + key,
            )
            raw[key] = a.astype(np.uint16)
        for key, a in packed(s, arrays).items():
            np.testing.assert_array_equal(raw[key], bits(a))
        np.testing.assert_array_equal(
            raw["progress"], np.tile([nt // 2, 1, epoch + 1], (rows, cols, 1))
        )
        if s["instrumentation"] == "sampled":
            tiles = [pack_tiles(v, rows, cols, "F") for v in products]
            expected = np.stack(
                [v.reshape(rows, cols, nt // 2, mt) for v in tiles], axis=3
            ).reshape(rows, cols, 2 * mt * nt)
            np.testing.assert_array_equal(raw["history"], bits(expected))
            words += rows * cols * 2 * mt * nt
        else:
            check(np.all(raw["history"] == 0), "pair rotation inactive observations")
        actual = unpack_tiles(raw["result"].view(np.float16), mt, nt, "F").astype(float)
        np.testing.assert_array_equal(bits(actual), bits(target))
        check(set(out) == {m["nodes"][-1]["host"]}, "pair rotation output port")
        np.testing.assert_array_equal(
            bits(out[m["nodes"][-1]["host"]]).ravel(), bits(actual).ravel()
        )
        numeric = accuracy(*arrays, s["pair_order"], actual)
        t = raw["timing"].astype(np.int64)
        cycles = sum(
            (t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)
        ) % (1 << 48)
        check(
            np.all((cycles > 0) & (cycles < 2**32)), "pair rotation bounded timestamps"
        )
        reports.append(
            dict(
                target_half_bits_exact=True,
                numerical=numeric,
                max_local_cycles=int(cycles.max()),
                cycles_per_pe=cycles.tolist(),
            )
        )
    report = dict(
        passed=True,
        profile=s["profile"],
        epochs=len(bs),
        actors=rows * cols,
        instrumentation=s["instrumentation"],
        internal_half_observations=words,
        cases=reports,
        performance_scope="Local WSE3 simulator interval, no global latency/hardware throughput claim",
    )
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
