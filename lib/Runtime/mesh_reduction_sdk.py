"""SDK collective-reduction host lifecycle and independent original-vector audit."""

import json, math
from pathlib import Path
import numpy as np
from frontend import check


def read(root, name):
    return json.loads((Path(root) / name).read_text())


def reference(operation, arrays):
    if operation == "dot":
        return math.fsum(float(x) * float(y) for x, y in zip(*arrays))
    return math.sqrt(math.fsum(float(x) * float(x) for x in arrays[0]))


def check_result(operation, arrays, result):
    ref = reference(operation, arrays)
    v = np.asarray(result)
    check(v.shape == (1,) and np.all(np.isfinite(v)), "scalar reduction output")
    np.testing.assert_allclose(v, [ref], rtol=3e-5, atol=3e-6)
    if operation == "nrm2" and ref != 0:
        np.testing.assert_allclose(v, [ref], rtol=3e-5, atol=0)
    if not any(arrays[0]):
        np.testing.assert_array_equal(v, [0.0])
    return dict(
        contract="fixed-original-vector-reduction-v1",
        fixed_accuracy_passed=True,
        max_abs_error=float(abs(v[0] - ref)),
    )


def distribute(values, s):
    count = s["rows"] * s["cols"] * s["local_length"]
    v = np.zeros(count, np.float32)
    v[: s["N"]] = values
    return v.reshape(s["rows"], s["cols"], s["local_length"])


def run(root):
    import os, subprocess
    from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder
    from mesh_common import sdk_runtime

    root = Path(root).resolve()
    os.chdir(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    h, w, l = s["rows"], s["cols"], s["local_length"]
    cmd = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        f"--fabric-dims={w+7},{h+2}",
        "--fabric-offsets=4,1",
        f"--params=rows:{h},cols:{w},N:{s['N']},local_length:{l},norm:{int(s['operation']=='nrm2')}",
        "-o=out",
        "--memcpy",
        "--channels=1",
    ]
    (root / "sdk-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    subprocess.run(cmd, check=True)

    def mark(label):
        (root / "runtime-stage.json").write_text(
            json.dumps(dict(operation=label)) + "\n"
        )
        print(label, flush=True)

    r = sdk_runtime(root)
    ids = {
        n: r.get_id(n) for n in ("x", "y", "result", "witness", "progress", "timing")
    }
    r.load()
    r.run()
    results = dict(success=False, runtime_instances=1, cases=[], diagnostics=[])

    def get(name, size, short=False):
        mark("D2H " + name)
        v = np.zeros(h * w * size, np.uint32 if short else np.float32)
        r.memcpy_d2h(
            v,
            ids[name],
            0,
            0,
            w,
            h,
            size,
            streaming=False,
            data_type=(
                MemcpyDataType.MEMCPY_16BIT if short else MemcpyDataType.MEMCPY_32BIT
            ),
            order=MemcpyOrder.COL_MAJOR,
            nonblock=False,
        )
        return v.reshape(h, w, size, order="F").tolist()

    for epoch, b in enumerate(read(root, "batches.json")):
        for name, n in zip(("x", "y"), m["nodes"][:-2]):
            mark("H2D " + name)
            v = distribute(b[n["host"]], s)
            r.memcpy_h2d(
                ids[name],
                v.ravel(order="F"),
                0,
                0,
                w,
                h,
                l,
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_32BIT,
                order=MemcpyOrder.COL_MAJOR,
                nonblock=False,
            )
        mark("launch main")
        r.launch("main", nonblock=False)
        replicas = get("result", 1)
        witness = get("witness", 4)
        progress = get("progress", 2, True)
        timing = get("timing", 6, True)
        results["cases"].append({m["nodes"][-1]["host"]: replicas[0][0]})
        results["diagnostics"].append(
            dict(replicas=replicas, witness=witness, progress=progress, timing=timing)
        )
        (root / "results.json").write_text(json.dumps(results) + "\n")
        print("REDUCTION EPOCH", epoch + 1, "COMPLETE", flush=True)
    r.stop()
    results["success"] = True
    (root / "results.json").write_text(json.dumps(results) + "\n")


def audit(root):
    from integrity import verify_bundle
    from mesh_reduction import plan, inputs, TEMPLATES

    root = Path(root)
    verify_bundle(root, implementation=False)
    m = read(root, "semantic.json")
    s = read(root, "schedule.json")
    result = read(root, "results.json")
    batches = read(root, "batches.json")
    check(s == plan(m), "reduction plan provenance")
    for name, t in TEMPLATES.items():
        check(
            (root / name).read_bytes()
            == (root / "implementation/runtime" / t).read_bytes(),
            "reduction generated source " + name,
        )
    check(
        result["success"]
        and result["runtime_instances"] == 1
        and len(result["cases"])
        == len(result["diagnostics"])
        == len(batches)
        == s["epochs"],
        "reduction lifecycle",
    )
    checks = []
    times = []
    l = s["local_length"]
    norm = s["operation"] == "nrm2"
    for epoch, (b, c, d) in enumerate(
        zip(batches, result["cases"], result["diagnostics"])
    ):
        arrays = inputs(m, b)
        out = c[m["nodes"][-1]["host"]]
        checks.append(check_result(s["operation"], arrays, out))
        replicas = np.asarray(d["replicas"], np.float32)
        np.testing.assert_array_equal(
            replicas, np.full((s["rows"], s["cols"], 1), out[0], np.float32)
        )
        np.testing.assert_array_equal(
            d["progress"],
            np.tile([2 if norm else 1, epoch + 1], (s["rows"], s["cols"], 1)),
        )
        witness = np.asarray(d["witness"])
        check(witness.shape == (s["rows"], s["cols"], 4), "reduction witness shape")
        peak = max(map(abs, arrays[0]))
        alpha = 1.0 if peak == 0 else 2.0 ** max(-126, math.frexp(peak)[1] - 1)
        for y in range(s["rows"]):
            for x in range(s["cols"]):
                start = (y * s["cols"] + x) * l
                end = min(start + l, s["N"])
                part = [v[start:end] for v in arrays]
                if norm:
                    localmax = max(map(abs, part[0]), default=0.0)
                    scaled = math.fsum((float(v) / alpha) ** 2 for v in part[0])
                    want = [localmax, peak, scaled, alpha]
                else:
                    want = [reference("dot", part), 0.0, 0.0, 0.0]
                np.testing.assert_allclose(witness[y, x], want, rtol=3e-5, atol=3e-6)
                if norm:
                    np.testing.assert_array_equal(
                        witness[y, x, [0, 1, 3]], np.asarray(want)[[0, 1, 3]]
                    )
        t = np.asarray(d["timing"])
        check(t.shape == (s["rows"], s["cols"], 6), "reduction timing shape")
        cycles = []
        for words in t.reshape(-1, 6):
            z = (
                sum(int(words[i + 3]) << (16 * i) for i in range(3))
                - sum(int(words[i]) << (16 * i) for i in range(3))
            ) % (1 << 48)
            check(0 < z < 1 << 32, "reduction local interval")
            cycles.append(z)
        times.append(max(cycles))
    return dict(
        passed=True,
        fixed_accuracy_passed=True,
        actors=s["rows"] * s["cols"],
        epochs=s["epochs"],
        vector_values=s["N"],
        numerical_checks=checks,
        max_local_cycles=times,
        timing_scope="Maximum local compute plus collective interval; excludes H2D/D2H, not a synchronized global or hardware interval",
        witness_scope="all-PE local dot or local max/global max/scaled-sum/scale; replicated scalar and callback counts",
    )
