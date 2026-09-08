"""SDK transport and source-ordered half reduction audits for replicated GEMV."""

import json, tempfile
from pathlib import Path
from frontend import check
from mesh_grouped_gemv import plan, generate, pack, reference
from half_matrix import inputs
from binary16 import matmul, roundoff, assert_bits_equal


def read(root, name):
    return json.loads((root / name).read_text())


def run(root):
    import os, subprocess, numpy as np
    from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder
    from cerebras.sdk.sdk_utils import input_array_to_u32
    from mesh_common import sdk_runtime

    root = Path(root).resolve()
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    p, mt, nt = s["P"], s["Mt"], s["Nt"]
    os.chdir(root)
    params = f"P:{p},Mt:{mt},Nt:{nt},pe_num_group:{s['group_size']},root_1st_phase:{s['root_within_group']},root_2nd_phase:{s['global_root']}"
    cmd = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        f"--fabric-dims={p+7},{p+2}",
        "--fabric-offsets=4,1",
        "--params=" + params,
        "-o=out",
        "--memcpy",
        "--channels=1",
        "--width-west-buf=0",
        "--width-east-buf=0",
    ]
    (root / "sdk-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    subprocess.run(cmd, check=True)
    runner = sdk_runtime(root)
    ids = {
        k: runner.get_id(k)
        for k in [
            "X",
            "W",
            "res",
            "hls_history",
            "hls_progress",
            "hls_timing",
            "hls_queue",
        ]
    }

    def get(name, n):
        v = np.zeros(p * p * n, np.uint32)
        runner.memcpy_d2h(
            v,
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
        return v.astype(np.uint16).reshape(p, p, n)

    r = dict(success=False, cases=[], diagnostics=[], runtime_instances=1, launches=[])

    def save():
        (root / "results.json").write_text(json.dumps(r) + "\n")

    runner.load()
    runner.run()
    try:
        for epoch, batch in enumerate(read(root, "batches.json")):
            for name, v in zip(["X", "W"], pack(*inputs(m, batch), p)):
                raw = input_array_to_u32(np.asarray(v, np.float16).ravel(), 1, 1)
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
            bits = get("res", nt)
            r["diagnostics"].append(
                dict(
                    result_bits=bits.tolist(),
                    history_bits=(
                        get("hls_history", 3 * nt).reshape(p, p, 3, nt).tolist()
                        if s["instrumentation"] == "sampled"
                        else None
                    ),
                    progress=get("hls_progress", 4).tolist(),
                    timing=get("hls_timing", 12).tolist(),
                    queue=get("hls_queue", 2).tolist(),
                )
            )
            v = bits[0].view(np.float16).astype(float).ravel()
            if not np.all(np.isfinite(v)):
                save()
                raise ValueError("nonfinite grouped half output")
            r["cases"].append({m["nodes"][3]["host"]: v.tolist()})
            save()
            print("GROUPED HALF EPOCH", epoch + 1, "COMPLETE", flush=True)
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
    check(s == plan(m), "grouped schedule regeneration")
    with tempfile.TemporaryDirectory() as td:
        generate(s, td)
        for name in ["pe.csl", "layout.csl", "grouped_comm.csl", "grouped_routes.csl"]:
            check(
                (root / name).read_bytes() == (Path(td) / name).read_bytes(),
                "grouped CSL regeneration " + name,
            )
    check(
        r["success"]
        and r["runtime_instances"] == 1
        and len(r["cases"]) == len(r["diagnostics"]) == len(batches) == m["epochs"]
        and r["launches"] == ["hls_main"] * len(batches),
        "grouped lifecycle",
    )
    p, mt, nt = s["P"], s["Mt"], s["Nt"]
    size = s["group_size"]
    reports = []
    observations = 0
    sampled = s["instrumentation"] == "sampled"

    def bits(v):
        return np.asarray(v, np.float16).view(np.uint16)

    for epoch, (batch, output, d) in enumerate(
        zip(batches, r["cases"], r["diagnostics"])
    ):
        a, b = inputs(m, batch)
        local, groups, total = reference(s, a, b)
        arrays = {}
        for key, shape in [
            ("result_bits", (p, p, nt)),
            ("history_bits", (p, p, 3, nt)),
            ("progress", (p, p, 4)),
            ("timing", (p, p, 12)),
            ("queue", (p, p, 2)),
        ]:
            if key == "history_bits" and not sampled:
                check(
                    d[key] is None, "grouped counter mode must not claim phase records"
                )
                continue
            v = np.asarray(d[key])
            check(
                v.shape == shape
                and np.issubdtype(v.dtype, np.integer)
                and np.all((v >= 0) & (v < 65536)),
                "grouped diagnostic shape/word " + key,
            )
            arrays[key] = v.astype(np.uint16)
        np.testing.assert_array_equal(
            arrays["progress"], np.tile([1, 1, 1, epoch + 1], (p, p, 1))
        )
        check(np.all((arrays["queue"] & 252) == 252), "grouped drained queues")
        if sampled:
            assert_bits_equal(
                arrays["history_bits"][:, :, 0],
                bits(local),
                f"epoch{epoch} local contraction [PErow,PEcolumn,lane]",
            )
            observations += p * p * nt
            for group in range(s["groups"]):
                y = group * size + s["root_within_group"]
                assert_bits_equal(
                    arrays["history_bits"][y, :, 1],
                    bits(groups[group]),
                    f"epoch{epoch} group{group} root-row{y} [PEcolumn,lane]",
                )
                observations += p * nt
            assert_bits_equal(
                arrays["history_bits"][s["global_root"], :, 2],
                bits(total),
                f"epoch{epoch} global-root row{s['global_root']} [PEcolumn,lane]",
            )
            observations += p * nt
        # Phase records at non-root PEs are inactive, not intermediate sums.
        for y in range(p):
            assert_bits_equal(
                arrays["result_bits"][y],
                bits(total),
                f"epoch{epoch} replica row{y} [PEcolumn,lane]",
            )
        result = np.asarray(output[m["nodes"][3]["host"]], float).reshape(1, s["N"])
        check(
            np.array_equal(result, result.astype(np.float16).astype(float)),
            "grouped logical half values",
        )
        np.testing.assert_array_equal(bits(result).ravel(), bits(total).ravel())
        ticks = []
        if not sampled:
            check(
                np.all(arrays["timing"][:, :, 6:] == 0),
                "grouped counters omit compute timing",
            )
        for offset in ((0, 6) if sampled else (0,)):
            t = arrays["timing"].astype(np.int64)
            v = sum(
                (t[:, :, offset + i + 3] - t[:, :, offset + i]) * (1 << (16 * i))
                for i in range(3)
            )
            check(np.all((v > 0) & (v < 2**32)), "grouped timestamp interval")
            ticks.append(v)
        native = matmul(a, b)
        reports.append(
            dict(
                final_bits_exact=True,
                active_phase_bits_exact=True if sampled else None,
                roundoff=roundoff(a, b, result),
                native_roundoff=roundoff(a, b, native),
                native_bits_equal=bool(np.array_equal(bits(result), bits(native))),
                max_native_abs_difference=float(np.max(np.abs(result - native))),
                max_local_cycles=int(ticks[0].max()),
                compute_cycles_median=float(np.median(ticks[1])) if sampled else None,
                total_cycles_per_pe=ticks[0].tolist(),
            )
        )
    report = dict(
        passed=True,
        profile=s["profile"],
        instrumentation=s["instrumentation"],
        phase_records_observed=sampled,
        compute_timing_observed=sampled,
        epochs=len(batches),
        actors=p * p,
        internal_half_observations=observations,
        cases=reports,
        contract="Exact local fused half contraction, active group/global root half addition records and every replicated final result; non-root phase records inactive; relaxed high-level order plus original-product bound",
    )
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
