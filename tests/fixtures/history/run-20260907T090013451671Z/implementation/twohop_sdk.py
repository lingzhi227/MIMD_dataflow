"""Half-typed SDK transport and bit-exact scheduled FMA/pointer-lifetime auditing."""

import json, tempfile
from pathlib import Path
from frontend import check
from mesh_twohop import plan, generate, pack, block_index
from half_matrix import inputs
from mesh_common import unpack_tiles
from binary16 import matmul, roundoff, assert_bits_equal


def read(root, name):
    return json.loads((root / name).read_text())


def run(root):
    import os, subprocess, numpy as np
    from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder
    from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view
    from mesh_common import sdk_runtime

    root = Path(root).resolve()
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    p, mt, kt, nt = (s[k] for k in ["P", "Mt", "Kt", "Nt"])
    os.chdir(root)
    cmd = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        f"--fabric-dims={p+7},{p+2}",
        "--fabric-offsets=4,1",
        f"--params=P:{p},Mt:{mt},Kt:{kt},Nt:{nt}",
        "-o=out",
        "--memcpy",
        "--channels=1",
        "--width-west-buf=0",
        "--width-east-buf=0",
    ]
    (root / "sdk-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    subprocess.run(cmd, check=True)
    runner = sdk_runtime(root)
    names = [
        "X",
        "W",
        "res",
        "hls_history",
        "hls_witness",
        "hls_timing",
        "hls_total_timing",
        "hls_progress",
        "hls_queue",
    ]
    ids = {k: runner.get_id(k) for k in names}

    def get(name, length):
        raw = np.zeros(p * p * length, np.uint32)
        runner.memcpy_d2h(
            raw,
            ids[name],
            0,
            0,
            p,
            p,
            length,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_16BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )
        return raw.astype(np.uint16).reshape(p, p, length)

    r = dict(success=False, cases=[], diagnostics=[], runtime_instances=1, launches=[])

    def save():
        (root / "results.json").write_text(json.dumps(r) + "\n")

    runner.load()
    runner.run()
    try:
        for epoch, batch in enumerate(read(root, "batches.json")):
            arrays = inputs(m, batch)
            for name, v in zip(["X", "W"], pack(*arrays, p)):
                raw = input_array_to_u32(np.asarray(v, dtype=np.float16).ravel(), 1, 1)
                runner.memcpy_h2d(
                    ids[name],
                    raw,
                    0,
                    0,
                    p,
                    p,
                    v.shape[-1],
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_16BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
            (root / "runtime-stage.json").write_text(
                json.dumps(dict(epoch=epoch, operation="launch hls_main"))
            )
            runner.launch("hls_main", nonblock=False)
            r["launches"].append("hls_main")
            bits = get("res", mt * nt)
            d = dict(
                result_bits=bits.tolist(),
                history_bits=(
                    get("hls_history", p * mt * nt).reshape(p, p, p, mt * nt).tolist()
                    if s["instrumentation"] == "sampled"
                    else None
                ),
                witness_bits=get("hls_witness", p * 4).reshape(p, p, p, 4).tolist(),
                timing=get("hls_timing", p * 6).reshape(p, p, p, 6).tolist(),
                total_timing=get("hls_total_timing", 6).tolist(),
                progress=get("hls_progress", 7).tolist(),
                queue=get("hls_queue", 2).tolist(),
            )
            r["diagnostics"].append(d)
            value = unpack_tiles(bits.view(np.float16), mt, nt, "F").astype(float)
            if not np.all(np.isfinite(value)):
                save()
                raise ValueError("nonfinite half output; raw diagnostic bits retained")
            r["cases"].append({m["nodes"][3]["host"]: value.ravel().tolist()})
            save()
            print("TWOHOP HALF EPOCH", epoch + 1, "COMPLETE", flush=True)
    finally:
        runner.stop()
    r["success"] = True
    save()


def audit(root):
    import numpy as np

    root = Path(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    batches = read(root, "batches.json")
    r = read(root, "results.json")
    check(s == plan(m), "two-hop schedule regeneration")
    with tempfile.TemporaryDirectory() as td:
        generate(s, td)
        for name in ["pe.csl", "layout.csl", "twohop_comm.csl", "twohop_routes.csl"]:
            check(
                (root / name).read_bytes() == (Path(td) / name).read_bytes(),
                "two-hop CSL regeneration " + name,
            )
    check(
        r["success"]
        and r["runtime_instances"] == 1
        and len(r["cases"]) == len(r["diagnostics"]) == len(batches) == m["epochs"]
        and r["launches"] == ["hls_main"] * len(batches),
        "two-hop lifecycle",
    )
    p, mt, kt, nt = (s[k] for k in ["P", "Mt", "Kt", "Nt"])
    sampled = s["instrumentation"] == "sampled"
    reports = []
    observations = 0
    for epoch, (b, o, d) in enumerate(zip(batches, r["cases"], r["diagnostics"])):
        a, bb = inputs(m, b)
        result = np.asarray(o[m["nodes"][3]["host"]], dtype=float).reshape(
            a.shape[0], bb.shape[1]
        )
        arrays = {}
        for key, shape in [
            ("result_bits", (p, p, mt * nt)),
            ("history_bits", (p, p, p, mt * nt)),
            ("witness_bits", (p, p, p, 4)),
            ("timing", (p, p, p, 6)),
            ("total_timing", (p, p, 6)),
            ("progress", (p, p, 7)),
            ("queue", (p, p, 2)),
        ]:
            if key == "history_bits" and not sampled:
                check(d[key] is None, "counter mode must not claim prefix observations")
                continue
            v = np.asarray(d[key])
            check(
                v.shape == shape
                and np.issubdtype(v.dtype, np.integer)
                and np.all((v >= 0) & (v < 65536)),
                "two-hop diagnostic shape/word " + key,
            )
            arrays[key] = v.astype(np.uint16)
        raw = unpack_tiles(arrays["result_bits"], mt, nt, "F")
        np.testing.assert_array_equal(result.astype(np.float16).view(np.uint16), raw)
        check(
            np.array_equal(result, result.astype(np.float16).astype(float)),
            "two-hop logical half output",
        )
        np.testing.assert_array_equal(
            arrays["progress"],
            np.tile([p, p, p, p, p // 2, p // 2 + 1, epoch + 1], (p, p, 1)),
        )
        check(np.all((arrays["queue"] & 60) == 60), "two-hop drained data queues")
        local_cycles = []
        for y in range(p):
            for x in range(p):
                acc = np.zeros((mt, nt), dtype=float)
                for step in range(p):
                    k = block_index(p, y, x, step)
                    left = a[y * mt : (y + 1) * mt, k * kt : (k + 1) * kt]
                    right = bb[k * kt : (k + 1) * kt, x * nt : (x + 1) * nt]
                    expected = np.asarray(
                        [left[0, 0], left[-1, -1], right[0, 0], right[-1, -1]],
                        np.float16,
                    ).view(np.uint16)
                    assert_bits_equal(
                        arrays["witness_bits"][y, x, step],
                        expected,
                        f"epoch{epoch} PE({x},{y}) round{step} Kblock{k} operand corners",
                    )
                    for j in range(kt):
                        acc = np.asarray(
                            acc + left[:, j, None] * right[None, j, :], np.float16
                        ).astype(float)
                    check(
                        np.all(np.isfinite(acc)), "two-hop finite scheduled recurrence"
                    )
                    bits = acc.astype(np.float16).view(np.uint16).ravel(order="F")
                    if sampled:
                        assert_bits_equal(
                            arrays["history_bits"][y, x, step],
                            bits,
                            f"epoch{epoch} PE({x},{y}) round{step} Kblock{k} column-major prefix",
                        )
                    words = arrays["timing"][y, x, step].astype(np.int64)
                    ticks = sum(
                        int(words[i + 3] - words[i]) * (1 << (16 * i)) for i in range(3)
                    )
                    check(0 < ticks < 2**32, "two-hop compute timestamp")
                    local_cycles.append(ticks)
                    observations += (mt * nt if sampled else 0) + 4
                assert_bits_equal(
                    arrays["result_bits"][y, x],
                    bits,
                    f"epoch{epoch} PE({x},{y}) final column-major tile",
                )
        t = arrays["total_timing"].astype(np.int64)
        total = sum((t[:, :, i + 3] - t[:, :, i]) * (1 << (16 * i)) for i in range(3))
        check(np.all((total > 0) & (total < 2**32)), "two-hop total timestamp")
        native = matmul(a, bb)
        reports.append(
            dict(
                scheduled_final_bits_exact=True,
                scheduled_prefix_bits_exact=True if sampled else None,
                roundoff=roundoff(a, bb, result),
                native_roundoff=roundoff(a, bb, native),
                native_bits_equal=bool(
                    np.array_equal(
                        result.astype(np.float16).view(np.uint16),
                        native.astype(np.float16).view(np.uint16),
                    )
                ),
                max_native_abs_difference=float(np.max(np.abs(result - native))),
                compute_cycles_median=float(np.median(local_cycles)),
                max_local_cycles=int(total.max()),
                total_cycles_per_pe=total.tolist(),
            )
        )
    report = dict(
        passed=True,
        profile=s["profile"],
        instrumentation=s["instrumentation"],
        prefix_observed=sampled,
        epochs=len(batches),
        actors=p * p,
        internal_half_observations=observations,
        cases=reports,
        contract="bit-exact scheduled binary16 FMA recurrence and exact block ownership; high-level/native relaxed block order may round differently; independent original-product forward bound",
    )
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
