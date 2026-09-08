"""SDK-native distributed FFT transport, lifecycle and original-transform audit."""

import json
from pathlib import Path
import numpy as np
from frontend import check
from mesh_fft import (
    inputs,
    pack,
    unpack,
    twiddles,
    interleave,
    reference,
    plan,
    TEMPLATES,
    source_text,
    phase_reference,
    phase_samples,
    logical_output,
)


def read(root, name):
    return json.loads((Path(root) / name).read_text())


def compiler_parameters(s):
    from compiler_parameters import encode

    return encode(
        dict(
            N="u16",
            P="i16",
            inverse="i16",
            normalization="i16",
            sampled="i16",
            restore="i16",
        ),
        dict(
            N=s["N"],
            P=s["rows"],
            inverse=int(s["transform"]["direction"] == "inverse"),
            normalization=["backward", "ortho", "forward"].index(
                s["transform"]["norm"]
            ),
            sampled=int(s["instrumentation"] == "sampled"),
            restore=int(s["result_layout"] == "input_layout"),
        ),
    )


def run(root):
    import os, subprocess
    from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder
    from mesh_common import sdk_runtime

    root = Path(root).resolve()
    os.chdir(root)
    s = read(root, "schedule.json")
    m = read(root, "semantic.json")
    p = s["rows"]
    n = s["N"]
    l = s["local_length"]
    cmd = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        f"--fabric-dims={p+7},{p+2}",
        "--fabric-offsets=4,1",
        compiler_parameters(s),
        "-o=out",
        "--memcpy",
        "--channels=1",
    ]
    (root / "sdk-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    subprocess.run(cmd, check=True)
    r = sdk_runtime(root)
    ids = {
        name: r.get_id(name)
        for name in (
            "X",
            "twiddle_array",
            "timing",
            "progress",
            "queues",
            "phase_samples",
            "phase_counts",
        )
    }
    r.load()
    r.run()

    def put(name, value, size):
        r.memcpy_h2d(
            ids[name],
            np.asarray(value, np.float32).ravel(),
            0,
            0,
            p,
            p,
            size,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )

    def get(name, size, short=False):
        v = np.zeros(p * p * size, np.uint32 if short else np.float32)
        r.memcpy_d2h(
            v,
            ids[name],
            0,
            0,
            p,
            p,
            size,
            streaming=False,
            data_type=(
                MemcpyDataType.MEMCPY_16BIT if short else MemcpyDataType.MEMCPY_32BIT
            ),
            order=MemcpyOrder.ROW_MAJOR,
            nonblock=False,
        )
        return v.reshape(p, p, size).tolist()

    result = dict(success=False, runtime_instances=1, cases=[], diagnostics=[])
    # Twiddle table is immutable for all calls. No arithmetic or transform on host.
    put("twiddle_array", np.broadcast_to(twiddles(n), (p, p, n)), n)
    for epoch, b in enumerate(read(root, "batches.json")):
        (root / "runtime-stage.json").write_text(
            json.dumps(dict(epoch=epoch, operation="H2D and FFT launch")) + "\n"
        )
        put("X", pack(inputs(m, b), s), l)
        r.launch("main", nonblock=False)
        packed = get("X", l)
        result["cases"].append(
            {s["output"]: interleave(logical_output(unpack(packed, s), s))}
        )
        result["diagnostics"].append(
            dict(
                packed_output=packed,
                phase_samples=(
                    get("phase_samples", s["phase_count"] * 4 * s["T"] ** 2)
                    if s["instrumentation"] == "sampled"
                    else None
                ),
                phase_counts=(
                    get("phase_counts", 7, True)
                    if s["instrumentation"] == "sampled"
                    else None
                ),
                timing=get("timing", 4, True),
                progress=get("progress", 2, True),
                queues=get("queues", 2, True),
            )
        )
        (root / "results.json").write_text(json.dumps(result) + "\n")
        print("FFT EPOCH", epoch + 1, "COMPLETE", flush=True)
    r.stop()
    result["success"] = True
    (root / "results.json").write_text(json.dumps(result) + "\n")


def numerical_check(x, y, transform):
    ref = reference(x, transform)
    check(y.shape == ref.shape and np.all(np.isfinite(y)), "FFT finite output shape")
    # Fixed normwise f32 screen; explicit near-zero scale derives from original transform.
    err = np.abs(y - ref)
    rn = float(np.linalg.norm(ref.ravel()))
    en = float(np.linalg.norm(err.ravel()))
    if rn == 0:
        check(np.array_equal(y, ref), "FFT exact zero reset")
    else:
        check(en / rn <= 2e-5, "FFT relative L2 accuracy")
        check(
            float(err.max()) <= 3e-5 * float(np.max(np.abs(ref))),
            "FFT max error relative to reference peak",
        )
    return dict(
        relative_l2_error=en / rn if rn else 0.0,
        max_abs_error=float(err.max()),
        fixed_accuracy_passed=True,
    )


def audit(root):
    from integrity import verify_bundle

    root = Path(root)
    verify_bundle(root, implementation=False)
    m = read(root, "semantic.json")
    s = read(root, "schedule.json")
    r = read(root, "results.json")
    b = read(root, "batches.json")
    check(plan(m) == s, "FFT schedule provenance")
    for name, t in TEMPLATES.items():
        check(
            (root / name).read_text() == source_text(s, name),
            "FFT generated source " + name,
        )
    check(
        r["success"]
        and r["runtime_instances"] == 1
        and len(r["cases"]) == len(r["diagnostics"]) == len(b) == m["epochs"],
        "FFT completed warm calls",
    )
    rows = []
    p = s["rows"]
    for epoch, (batch, case, d) in enumerate(zip(b, r["cases"], r["diagnostics"])):
        packed = np.asarray(d["packed_output"], np.float32)
        check(packed.shape == (p, p, s["local_length"]), "FFT packed extent")
        physical = unpack(packed, s)
        y = logical_output(physical, s)
        check(case == {s["output"]: interleave(y)}, "FFT final host layout consistency")
        x = inputs(m, batch)
        v = numerical_check(x, y, s["transform"])
        if s["instrumentation"] == "sampled":
            sample = np.asarray(d["phase_samples"], np.float32)
            count = np.asarray(d["phase_counts"])
            check(
                sample.shape == (p, p, s["phase_count"] * 4 * s["T"] ** 2)
                and np.all(np.isfinite(sample)),
                "FFT internal sample shape/finite",
            )
            check(
                count.shape == (p, p, 7)
                and np.all(count[:, :, : s["phase_count"]] == 1)
                and np.all(count[:, :, s["phase_count"] :] == 0),
                "FFT active stage callbacks once and omitted stages zero per warm epoch",
            )
            sample = sample.reshape(p, p, s["phase_count"], -1)
            errors = []
            for phase, ref in enumerate(phase_reference(x, s)):
                expected = phase_samples(ref, s)
                err = np.abs(sample[:, :, phase, :].astype(np.float64) - expected)
                tolerance = 3e-5 * float(np.max(np.abs(ref)))
                check(
                    float(err.max()) <= tolerance,
                    f"FFT internal phase{phase} pencil endpoints mismatch, max error{err.max()} tolerance{tolerance}",
                )
                errors.append(float(err.max()))
            check(
                np.array_equal(
                    sample[:, :, s["phase_count"] - 1, :], phase_samples(physical, s)
                ),
                "FFT final sample/result consistency",
            )
            v["phase_max_abs_errors"] = errors
        else:
            check(
                d["phase_samples"] is None and d["phase_counts"] is None,
                "FFT unobserved phases explicitly null",
            )
        progress = np.asarray(d["progress"])
        q = np.asarray(d["queues"])
        time = np.asarray(d["timing"])
        check(
            progress.shape == (p, p, 2) and np.all(progress == epoch + 1),
            "FFT entry/callback epoch count",
        )
        check(
            q.shape == (p, p, 2) and np.all(q == q.astype(np.uint16)),
            "FFT queue witness shape/type",
        )
        check(
            np.all((q[:, :, 0].astype(np.uint16) & 20) == 20)
            and np.all((q[:, :, 1].astype(np.uint16) & 40) == 40),
            "FFT owned queues drained",
        )
        check(
            time.shape == (p, p, 4) and np.all(time == time.astype(np.uint16)),
            "FFT timestamp limbs",
        )
        cycles = (
            time[:, :, 0].astype(np.uint64)
            + (time[:, :, 1].astype(np.uint64) << 16)
            + (time[:, :, 2].astype(np.uint64) << 32)
        )
        check(np.all(cycles > 0), "FFT positive local interval")
        v.update(max_local_cycles=int(cycles.max()), cycles_per_pe=cycles.tolist())
        rows.append(v)
    return dict(
        passed=True,
        profile=s["profile"],
        epochs=len(b),
        actors=p * p,
        cases=rows,
        internal_phases_observed=s["instrumentation"] == "sampled",
        internal_float_observations=(
            p * p * s["phase_count"] * 4 * s["T"] ** 2 * len(b)
            if s["instrumentation"] == "sampled"
            else 0
        ),
        contract="Complete original C2C transform, declared pencil ownership, warm entry/callback and owned-queue checks. Optional per-stage pencil-endpoint samples explicitly scoped by the selected schedule and instrumentation; not full internal tensors. Fixed f32 normwise accuracy; simulator local intervals, not global/hardware latency.",
    )
