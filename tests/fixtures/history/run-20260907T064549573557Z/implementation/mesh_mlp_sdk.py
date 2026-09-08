"""SDK transport and staged target/protocol audit for rectangular resident MLP."""

import json, tempfile
from pathlib import Path
import numpy as np
from frontend import check
from mesh_mlp import plan, generate, inputs, reference, accuracy
from mesh_common import pack_tiles, unpack_tiles
from mesh_twohop import block_index, cycle
from compiler_parameters import encode


def read(root, name):
    return json.loads((Path(root) / name).read_text())


def save(root, r):
    tmp = root / "results.json.tmp"
    tmp.write_text(json.dumps(r) + "\n")
    tmp.replace(root / "results.json")


def parameters(s):
    p = s["P"]
    v = dict(
        P=p,
        dim_p_pe=s["Nt"],
        pes_p_head=p,
        pes_p_kv_head=p,
        head_dim_p_pe=s["Nt"],
        seq_len_p_pe=s["Mt"],
        ffn_dim_p_pe=s["Ft"],
        sampled=int(s["instrumentation"] == "sampled"),
    )
    return encode({k: "i16" for k in v}, v)


def packed(s, arrays):
    p = s["P"]
    out = {"x": pack_tiles(arrays[0], p, p, "F")}
    for name, a in zip(("up_weight", "gate_weight", "down_weight"), arrays[1:]):
        kt, nt = a.shape[0] // p, a.shape[1] // p
        out[name] = np.asarray(
            [
                [
                    a[
                        block_index(p, y, x, 0)
                        * kt : (block_index(p, y, x, 0) + 1)
                        * kt,
                        x * nt : (x + 1) * nt,
                    ].ravel(order="C")
                    for x in range(p)
                ]
                for y in range(p)
            ]
        )
    return out


def extents(s):
    p, L, H, W = [s[k] for k in ("P", "length", "hidden_length", "weight_length")]
    sample = s["instrumentation"] == "sampled"
    result = dict(
        x=L,
        up_weight=W,
        gate_weight=W,
        down_weight=W,
        result=L,
        up_history=p * H if sample else 1,
        gate_history=p * H if sample else 1,
        down_history=p * L if sample else 1,
        gate_snapshot=H if sample else 1,
        hidden_snapshot=H if sample else 1,
        activated_gate=H if sample else 1,
        left_first=2 * L + H if sample else 1,
        right_first=3 * W if sample else 1,
        progress=8,
        timing=6,
        queues=2,
    )

    if s.get("down_accumulation") == "block_f32":
        result["wide_accumulator"] = L
    return result


def run(root):
    import os, subprocess, time
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
    r = dict(
        success=False,
        runtime_instances=1,
        cases=[],
        diagnostics=[],
        launches=[],
        host_call_seconds=[],
        host_timing_scope="Host elapsed per complete input-transfer/launch/output-transfer call; excludes compilation and runtime load; not device throughput",
    )

    def stage(operation, port=None):
        (root / "runtime-stage.json").write_text(
            json.dumps(dict(epoch=len(r["cases"]), operation=operation, port=port))
            + "\n"
        )

    try:
        stage("initialize")
        runner.launch("init_task", nonblock=False)
        for b in read(root, "batches.json"):
            call_started = time.monotonic()
            for name, a in packed(s, inputs(m, b)).items():
                stage("host_to_device", name)
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
            stage("launch")
            runner.launch("hls_main", nonblock=False)
            r["launches"].append("hls_main")
            d = {}
            for name, n in extents(s).items():
                raw = np.zeros(rows * cols * n, np.uint32)
                stage("device_to_host", name)
                runner.memcpy_d2h(
                    raw,
                    ids[name],
                    0,
                    0,
                    cols,
                    rows,
                    n,
                    streaming=False,
                    data_type=(
                        MemcpyDataType.MEMCPY_32BIT
                        if name == "wide_accumulator"
                        else MemcpyDataType.MEMCPY_16BIT
                    ),
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
                d[name] = (
                    raw.astype(np.uint32 if name == "wide_accumulator" else np.uint16)
                    .reshape(rows, cols, n)
                    .tolist()
                )
            value = unpack_tiles(
                np.asarray(d["result"], np.uint16).view(np.float16),
                s["Mt"],
                s["Nt"],
                "F",
            ).astype(float)
            r["cases"].append({m["nodes"][-1]["host"]: value.ravel().tolist()})
            r["diagnostics"].append(d)
            r["host_call_seconds"].append(time.monotonic() - call_started)
            save(root, r)
            print("RECTANGULAR MLP HLS", len(r["cases"]), flush=True)
    finally:
        runner.stop()
    stage("completed")
    r["success"] = True
    save(root, r)


def audit(root):
    from integrity import verify_bundle, verify_codegen

    root = Path(root)
    verify_bundle(root)
    verify_codegen(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    check(s == plan(m), "MLP schedule regeneration")
    report = audit_cases(s, m, read(root, "batches.json"), read(root, "results.json"))
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def audit_cases(s, m, bs, r, require_complete=True):
    completed = len(r["cases"])
    check(
        r["runtime_instances"] == 1
        and len(bs) == m["epochs"]
        and len(r["diagnostics"]) == completed
        and r["launches"] == ["hls_main"] * completed
        and 1 <= completed <= m["epochs"]
        and (not r["success"] or completed == m["epochs"]),
        "MLP lifecycle",
    )
    complete = bool(r["success"] and completed == m["epochs"])
    if require_complete:
        check(complete, "MLP complete execution required")
    p = s["P"]
    ring = cycle(p)
    words = 0
    reports = []
    bits = lambda x: np.asarray(x, np.float16).view(np.uint16)
    for epoch, (b, d, o) in enumerate(zip(bs, r["diagnostics"], r["cases"])):
        arrays = inputs(m, b)
        target, witnesses = reference(s, *arrays)
        raw = {}
        check(set(d) == set(extents(s)), "MLP observed ports")
        for key, n in extents(s).items():
            a = np.asarray(d[key])
            check(
                a.shape == (p, p, n)
                and np.issubdtype(a.dtype, np.integer)
                and np.all(
                    (a >= 0) & (a < (2**32 if key == "wide_accumulator" else 65536))
                ),
                "MLP raw words " + key,
            )
            raw[key] = a.astype(np.uint32 if key == "wide_accumulator" else np.uint16)
        for key, a in packed(s, arrays).items():
            np.testing.assert_array_equal(raw[key], bits(a))
        for y in range(p):
            for x in range(p):
                np.testing.assert_array_equal(
                    raw["progress"][y, x],
                    [
                        (-ring.index(y)) % p,
                        (-ring.index(y)) % p,
                        p,
                        p,
                        p,
                        1,
                        epoch + 1,
                        3,
                    ],
                )
        check(np.all((raw["queues"] & 248) == 248), "MLP queues drained")
        for key, w in witnesses.items():
            if key == "wide_accumulator":
                np.testing.assert_array_equal(raw[key], w)
            elif s["instrumentation"] == "sampled":
                np.testing.assert_array_equal(raw[key], bits(w))
                words += w.size
            else:
                check(np.all(raw[key] == 0), "MLP inactive observation " + key)
        actual = unpack_tiles(
            raw["result"].view(np.float16), s["Mt"], s["Nt"], "F"
        ).astype(float)
        np.testing.assert_array_equal(bits(actual), bits(target))
        check(set(o) == {m["nodes"][-1]["host"]}, "MLP output port")
        np.testing.assert_array_equal(
            np.asarray(o[m["nodes"][-1]["host"]], float), actual.ravel()
        )
        numeric = accuracy(s, *arrays, actual)
        t = raw["timing"].astype(np.int64)
        cycles = sum(
            (t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)
        ) % (1 << 48)
        check(np.all((cycles > 0) & (cycles < 2**32)), "MLP timestamps")
        reports.append(
            dict(
                target_half_bits_exact=True,
                numerical=numeric,
                max_local_cycles=int(cycles.max()),
                cycles_per_pe=cycles.tolist(),
            )
        )
    return dict(
        passed=complete,
        complete=complete,
        completed_calls_valid=True,
        profile=s["profile"],
        epochs=completed,
        actors=p * p,
        instrumentation=s["instrumentation"],
        internal_half_observations=words,
        observed_f32_accumulator_values=(
            completed * p * p * s["length"]
            if s.get("down_accumulation") == "block_f32"
            else 0
        ),
        cases=reports,
        performance_scope="WSE3 simulator local intervals, supplied bounded X/U/G/D, no RMS/residual/full-model/hardware claim",
    )
