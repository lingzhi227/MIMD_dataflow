"""SDK device-aligned matmul transport and independent source-schedule state audit."""

import json, tempfile
from pathlib import Path
import numpy as np
from frontend import check
from mesh_device_matmul import plan, generate, inputs, reference, accuracy
from mesh_common import pack_tiles, unpack_tiles
from compiler_parameters import encode


def read(root, name):
    return json.loads((Path(root) / name).read_text())


def save(root, r):
    tmp = root / "results.json.tmp"
    tmp.write_text(json.dumps(r) + "\n")
    tmp.replace(root / "results.json")


def parameters(s):
    p, mt, nt = s["P"], s["Mt"], s["Nt"]
    v = dict(
        P=p,
        dim_p_pe=nt,
        pes_p_head=p,
        pes_p_kv_head=p,
        head_dim_p_pe=nt,
        seq_len_p_pe=mt,
        ffn_dim_p_pe=nt,
        sampled=int(s["instrumentation"] == "sampled"),
    )
    return encode({k: "i16" for k in v}, v)


def packed(s, arrays):
    return {
        key: pack_tiles(a, s["P"], s["P"], "F") for key, a in zip(("a", "b"), arrays)
    }


def extents(s):
    p, L, S = s["P"], s["length"], s["left_length"]
    sample = s["instrumentation"] == "sampled"
    return dict(
        a=S,
        b=L,
        result=L,
        history=p * L if sample else 1,
        left_owners=p * S if sample else 1,
        right_owners=p * L if sample else 1,
        progress=4,
        timing=6,
        queues=2,
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
        runner.launch("init_task", nonblock=False)
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
            save(root, r)
            print("DEVICE MATMUL HLS", len(r["cases"]), flush=True)
    finally:
        runner.stop()
    r["success"] = True
    save(root, r)


def audit(root):
    from integrity import verify_bundle

    root = Path(root)
    verify_bundle(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    bs = read(root, "batches.json")
    r = read(root, "results.json")
    check(s == plan(m), "device matmul schedule regeneration")
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
                "device matmul regenerated " + name,
            )
    check(
        r["success"]
        and r["runtime_instances"] == 1
        and len(r["cases"]) == len(r["diagnostics"]) == len(bs) == m["epochs"]
        and r["launches"] == ["hls_main"] * len(bs),
        "device matmul lifecycle",
    )
    p, mt = s["P"], s["Mt"]
    bits = lambda x: np.asarray(x, np.float16).view(np.uint16)
    reports = []
    words = 0
    for epoch, (b, d, o) in enumerate(zip(bs, r["diagnostics"], r["cases"])):
        arrays = inputs(m, b)
        partial, left, right, target = reference(s, *arrays)
        raw = {}
        check(set(d) == set(extents(s)), "device matmul observed ports")
        for key, n in extents(s).items():
            a = np.asarray(d[key])
            check(
                a.shape == (p, p, n)
                and np.issubdtype(a.dtype, np.integer)
                and np.all((a >= 0) & (a < 65536)),
                "device matmul raw words " + key,
            )
            raw[key] = a.astype(np.uint16)
        for key, a in packed(s, arrays).items():
            np.testing.assert_array_equal(raw[key], bits(a))
        from mesh_twohop import cycle

        order = cycle(p)
        expected_progress = np.asarray(
            [
                [
                    [(-order.index(x)) % p, (-order.index(y)) % p, p, epoch + 1]
                    for x in range(p)
                ]
                for y in range(p)
            ],
            np.uint16,
        )
        np.testing.assert_array_equal(raw["progress"], expected_progress)
        check(
            np.all((raw["queues"] & 248) == 248), "device matmul owned queues drained"
        )
        if s["instrumentation"] == "sampled":
            np.testing.assert_array_equal(raw["history"], bits(partial))
            np.testing.assert_array_equal(raw["left_owners"], bits(left))
            np.testing.assert_array_equal(raw["right_owners"], bits(right))
            words += partial.size + left.size + right.size
        else:
            check(
                all(
                    np.all(raw[key] == 0)
                    for key in ("history", "left_owners", "right_owners")
                ),
                "device matmul inactive tensor observations",
            )
        actual = unpack_tiles(raw["result"].view(np.float16), mt, s["Nt"], "F").astype(
            float
        )
        np.testing.assert_array_equal(bits(actual), bits(target))
        check(set(o) == {m["nodes"][-1]["host"]}, "device matmul output port")
        np.testing.assert_array_equal(
            bits(o[m["nodes"][-1]["host"]]).ravel(), bits(actual).ravel()
        )
        numeric = accuracy(*arrays, actual)
        t = raw["timing"].astype(np.int64)
        cycles = sum(
            (t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)
        ) % (1 << 48)
        check(
            np.all((cycles > 0) & (cycles < 2**32)), "device matmul bounded timestamps"
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
        actors=p * p,
        instrumentation=s["instrumentation"],
        internal_half_observations=words,
        cases=reports,
        performance_scope="Local WSE3 simulator intervals, no full attention/hardware claim",
    )
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
