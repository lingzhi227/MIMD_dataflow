"""SDK score transport and independent source-schedule state audit."""

import json, tempfile
from pathlib import Path
import numpy as np
from frontend import check
from mesh_attention import plan, generate, inputs, reference, accuracy
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
    v["scale_bits"] = int(np.asarray(s["scale"], np.float16).view(np.uint16))
    types = {k: "i16" for k in v}
    types["scale_bits"] = "u16"
    return encode(types, v)


def packed(s, arrays):
    return {
        key: pack_tiles(a, s["P"], s["P"], "F")
        for key, a in zip(("q", "k", "v"), arrays)
    }


def extents(s):
    p, L, S = s["P"], s["length"], s["score_length"]
    sample = s["instrumentation"] == "sampled"
    return dict(
        q=L,
        k=L,
        result=L,
        v=L,
        probability=S if sample else 1,
        value_history=p * L if sample else 1,
        value_left=p * S if sample else 1,
        value_right=p * L if sample else 1,
        value_progress=4,
        history=p * S if sample else 1,
        owners=p * L if sample else 1,
        roots=p,
        progress=3,
        timing=6,
        queues=2,
        logits=S if sample else 1,
        exponents=S if sample else 1,
        softmax_history=5 * s["Mt"] if sample else 1,
        softmax_progress=6,
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
            print("RESIDENT ATTENTION HLS", len(r["cases"]), flush=True)
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
    check(s == plan(m), "score schedule regeneration")
    with tempfile.TemporaryDirectory() as td:
        generate(s, td)
        for name in (
            "layout.csl",
            "pe.csl",
            "inference_comm.csl",
            "inference_routes.csl",
            "WaferLLM-LICENSE.txt",
            "SOURCE-NOTICE.txt",
            "softmax_local.csl",
        ):
            check(
                (root / name).read_bytes() == (Path(td) / name).read_bytes(),
                "score regenerated " + name,
            )
    report = audit_cases(s, m, bs, r)
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def audit_cases(s, m, bs, r, require_complete=True):
    """Pure observed-prefix checks; partial diagnostics never qualify a bundle."""
    completed = len(r["cases"])
    check(
        r["runtime_instances"] == 1
        and len(bs) == m["epochs"]
        and len(r["diagnostics"]) == completed
        and r["launches"] == ["hls_main"] * completed
        and 1 <= completed <= m["epochs"]
        and (not r["success"] or completed == m["epochs"]),
        "attention observed-prefix lifecycle",
    )
    complete = bool(r["success"] and completed == m["epochs"])
    if require_complete:
        check(complete, "attention full lifecycle required")
    p, mt = s["P"], s["Mt"]
    bits = lambda x: np.asarray(x, np.float16).view(np.uint16)
    reports = []
    words = 0
    for epoch, (b, d, o) in enumerate(zip(bs, r["diagnostics"], r["cases"])):
        arrays = inputs(m, b)
        (
            partial,
            owners,
            roots,
            logits,
            soft_history,
            exponents,
            probability,
            value_history,
            value_left,
            value_right,
            target,
        ) = reference(s, *arrays)
        raw = {}
        check(set(d) == set(extents(s)), "score observed ports")
        for key, n in extents(s).items():
            a = np.asarray(d[key])
            check(
                a.shape == (p, p, n)
                and np.issubdtype(a.dtype, np.integer)
                and np.all((a >= 0) & (a < 65536)),
                "score raw words " + key,
            )
            raw[key] = a.astype(np.uint16)
        for key, a in packed(s, arrays).items():
            np.testing.assert_array_equal(raw[key], bits(a))
        np.testing.assert_array_equal(
            raw["progress"], np.tile([p, 1, epoch + 1], (p, p, 1))
        )
        from mesh_twohop import cycle

        ring = cycle(p)
        for y in range(p):
            for x in range(p):
                np.testing.assert_array_equal(
                    raw["value_progress"][y, x],
                    [(-ring.index(x)) % p, (-ring.index(y)) % p, p, epoch + 1],
                )
        np.testing.assert_array_equal(raw["roots"], roots)
        np.testing.assert_array_equal(
            raw["softmax_progress"], np.ones((p, p, 6), np.uint16)
        )
        check(np.all((raw["queues"] & 248) == 248), "score owned queues drained")
        if s["instrumentation"] == "sampled":
            for key, expected in [
                ("probability", pack_tiles(probability, p, p, "F")),
                ("value_history", value_history),
                ("value_left", value_left),
                ("value_right", value_right),
            ]:
                np.testing.assert_array_equal(raw[key], bits(expected))
                words += np.asarray(expected).size
            np.testing.assert_array_equal(raw["history"], bits(partial))
            np.testing.assert_array_equal(raw["owners"], bits(owners))
            np.testing.assert_array_equal(
                raw["logits"], bits(pack_tiles(logits, p, p, "F"))
            )
            np.testing.assert_array_equal(
                raw["exponents"], bits(pack_tiles(exponents, p, p, "F"))
            )
            np.testing.assert_array_equal(
                raw["softmax_history"], bits(soft_history.reshape(p, p, -1))
            )
            words += (
                partial.size
                + owners.size
                + logits.size
                + exponents.size
                + soft_history.size
            )
        else:
            check(
                all(
                    np.all(raw[key] == 0)
                    for key in (
                        "probability",
                        "value_history",
                        "value_left",
                        "value_right",
                        "history",
                        "owners",
                        "logits",
                        "exponents",
                        "softmax_history",
                    )
                ),
                "score inactive tensor observations",
            )
        actual = unpack_tiles(raw["result"].view(np.float16), mt, s["Nt"], "F").astype(
            float
        )
        np.testing.assert_array_equal(bits(actual), bits(target))
        check(set(o) == {m["nodes"][-1]["host"]}, "score output port")
        np.testing.assert_array_equal(
            bits(o[m["nodes"][-1]["host"]]).ravel(), bits(actual).ravel()
        )
        numeric = accuracy(s, *arrays, actual)
        t = raw["timing"].astype(np.int64)
        cycles = sum(
            (t[:, :, j + 3] - t[:, :, j]) * (1 << (16 * j)) for j in range(3)
        ) % (1 << 48)
        check(np.all((cycles > 0) & (cycles < 2**32)), "score bounded timestamps")
        reports.append(
            dict(
                target_half_bits_exact=True,
                numerical=numeric,
                max_local_cycles=int(cycles.max()),
                cycles_per_pe=cycles.tolist(),
            )
        )
    report = dict(
        passed=complete,
        completed_calls_valid=True,
        complete=complete,
        scope=(
            "complete_bundle_audit"
            if require_complete
            else "completed_call_prefix_diagnostic"
        ),
        profile=s["profile"],
        epochs=completed,
        actors=p * p,
        instrumentation=s["instrumentation"],
        internal_half_observations=words,
        cases=reports,
        performance_scope="Local WSE3 simulator intervals, supplied-Q/K/V unmasked single-head path; no full model/hardware claim",
    )
    return report
