"""Typed C2C pencil FFT using the SDK's native three-transform/four-transpose engine."""

import copy
from pathlib import Path
import numpy as np
from frontend import check

POLICY = dict(
    partition="pencils",
    exchange="sdk_transpose",
    compute="sdk_fft",
    result="input_layout",
    fp="relaxed",
)


def verify(module, epochs, bound):
    m = copy.deepcopy(module)
    check(
        [n["op"] for n in m["nodes"]] == ["input", "fft3d", "output"],
        "FFT single typed C2C transform",
    )
    a, op, out = m["nodes"]
    n = op["fft"]["N"]
    check(
        not m["states"]
        and not any("place" in x or x.get("dtype", "f32") != "f32" for x in m["nodes"]),
        "FFT owns f32 region",
    )
    check(
        op["inputs"] == [a["id"]]
        and out["inputs"] == [op["id"]]
        and len({x["id"] for x in m["nodes"]}) == 3,
        "FFT dependencies",
    )
    check(
        a["shape"] == op["shape"] == [n * n, 2 * n],
        "interleaved [y,x,complex z] FFT shape",
    )
    check(
        type(epochs) is int
        and 1 <= epochs <= 16
        and type(bound) is int
        and 0 <= bound <= 2147483647,
        "FFT execution bounds",
    )
    check(
        op["fft"]["direction"] in ("forward", "inverse")
        and op["fft"]["norm"] in ("backward", "ortho", "forward"),
        "FFT transform semantics",
    )
    out["shape"] = op["shape"][:]
    for x in m["nodes"]:
        x["interval"] = None
    m.update(profile="mesh_fft.v1", epochs=epochs, input_bound=bound)
    plan(m)
    return m


def plan(m, partitions=1):
    a, op, out = m["nodes"]
    d = op["dataflow"]
    n = op["fft"]["N"]
    p = d["rows"]
    check(
        set(d) == set(POLICY) | {"rows", "cols"}
        and all(d[k] == v for k, v in POLICY.items() if k != "result")
        and d["result"] in ("input_layout", "transposed_pencils"),
        "FFT supported SDK pencil policy",
    )
    check(
        partitions == 1 and type(n) is int and 4 <= n <= 64 and n & (n - 1) == 0,
        "FFT N power of two4..64",
    )
    check(
        type(p) is int
        and type(d["cols"]) is int
        and p == d["cols"]
        and 2 <= p <= 16
        and p & (p - 1) == 0
        and n % p == 0,
        "FFT square divisible power-of-two region",
    )
    t = n // p
    l = 2 * t * t * n
    restore = d["result"] == "input_layout"
    stage_count = 7 if restore else 5
    mode = m.get("instrumentation", "counters")
    check(mode in ("sampled", "counters"), "FFT instrumentation mode")
    memory = dict(
        phase_samples=stage_count * 4 * t * t * 4 if mode == "sampled" else 4,
        data=4 * l,
        transpose_workspace=4 * l,
        twiddles_and_workspaces=20 * n,
        protocol=128,
        sdk_code_stack_reserve=24576,
    )
    check(
        l <= 32767 and 2 * t * t * t <= 32767 and sum(memory.values()) <= 49152,
        "FFT signed descriptors and SRAM budget",
    )
    stages = [
        dict(kind="local_fft", axis="z"),
        dict(kind="transpose", vertical=False, next_vertical=True),
        dict(kind="local_fft", axis="x"),
        dict(kind="transpose", vertical=True, next_vertical=True),
        dict(kind="local_fft", axis="y"),
    ]
    if restore:
        stages += [
            dict(kind="transpose", vertical=True, next_vertical=False),
            dict(kind="transpose", vertical=False, next_vertical=False),
        ]
    return dict(
        result_layout=d["result"],
        output_axes=["y", "x", "z"] if restore else ["x", "z", "y"],
        phase_count=stage_count,
        profile=m["profile"],
        rows=p,
        cols=p,
        N=n,
        T=t,
        local_length=l,
        epochs=m["epochs"],
        transform=op["fft"],
        input=a["host"],
        output=out["host"],
        nodes=m["nodes"],
        memory=memory,
        stages=stages,
        resources=dict(
            colors=[4, 5, 6, 7],
            active_transpose_colors=[4, 5],
            color_contract="four SDK API slots conservatively reserved; transpose uses first two exclusively across dynamic route/teardown phases",
            control_wavelets=["SWITCH_ADV", "TEARDOWN"],
            input_queues=[2, 4],
            output_queues=[3, 5],
            local_tasks=[11, 12, 13],
            microthreads=[2, 3],
            dsr_banks=dict(src0=[1, 2, 3], src1=[1, 2, 3], dest=[1, 2, 3]),
            xdsr=[0],
            scalar_registers=[0, 1, 2],
            concurrent_composition=False,
        ),
        lifecycle="in-place SDK callback completion before buffer reuse or next launch; failed call requires runtime recreation",
        internal_phases_observed=mode == "sampled",
        instrumentation=mode,
    )


def inputs(m, b):
    a = m["nodes"][0]
    n = m["nodes"][1]["fft"]["N"]
    check(set(b) == {a["host"]}, "FFT input ports")
    x = np.asarray(b[a["host"]], np.float64)
    check(
        x.shape == (2 * n**3,)
        and np.all(np.isfinite(x))
        and np.max(np.abs(x)) <= m["input_bound"],
        "FFT finite bounded input extent",
    )
    x = x.astype(np.float32).reshape(n, n, n, 2)
    return (
        np.ascontiguousarray(x)
        .view(np.complex64)
        .reshape(n, n, n)
        .astype(np.complex128)
    )


def interleave(x):
    return np.stack((x.real, x.imag), axis=-1).astype(np.float32).ravel().tolist()


def reference(x, transform):
    fn = np.fft.ifftn if transform["direction"] == "inverse" else np.fft.fftn
    return fn(x, norm=transform["norm"])


def evaluate(m, batches):
    check(len(batches) == m["epochs"], "FFT epoch count")
    return (
        [
            {
                m["nodes"][-1]["host"]: interleave(
                    reference(inputs(m, b), m["nodes"][1]["fft"])
                )
            }
            for b in batches
        ],
        {},
    )


def pack(x, s):
    # SDK benchmark offset = z*T*T + (y%T)*T + x%T.
    n, p, t = s["N"], s["rows"], s["T"]
    return (
        np.asarray(interleave(x), np.float32)
        .reshape(p, t, p, t, n, 2)
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(p, p, -1)
    )


def unpack(x, s):
    n, p, t = s["N"], s["rows"], s["T"]
    a = (
        np.asarray(x, np.float32)
        .reshape(p, p, n, t, t, 2)
        .transpose(0, 3, 1, 4, 2, 5)
        .reshape(n, n, n, 2)
    )
    return (
        np.ascontiguousarray(a)
        .view(np.complex64)
        .reshape(n, n, n)
        .astype(np.complex128)
    )


def logical_output(physical, s):
    return (
        physical
        if s["result_layout"] == "input_layout"
        else physical.transpose(2, 0, 1)
    )


def twiddles(n):
    a = 2 * np.pi * np.arange(n // 2) / n
    return np.stack((np.cos(a), np.sin(a)), axis=-1).astype(np.float32).ravel()


def phase_reference(x, s):
    """Logical storage-axis transforms and permutations from the actual SDK driver."""
    phases = []
    v = x
    fn = np.fft.ifft if s["transform"]["direction"] == "inverse" else np.fft.fft
    for stage in s["stages"]:
        if stage["kind"] == "local_fft":
            v = fn(v, axis=2, norm=s["transform"]["norm"])
        elif stage["vertical"]:
            v = v.transpose(2, 1, 0)
        else:
            v = v.transpose(0, 2, 1)
        phases.append(v)
    return phases


def phase_samples(v, s):
    p = s["rows"]
    t = s["T"]
    n = s["N"]
    a = pack(v, s).reshape(p, p, n, t * t, 2)
    return np.concatenate((a[:, :, 0, :, :], a[:, :, -1, :, :]), axis=-1).reshape(
        p, p, 4 * t * t
    )


TEMPLATES = {
    "layout.csl": "fft_layout.csl",
    "pe.csl": "fft_pe.csl",
    "fft_driver.csl": "fft_driver.csl",
}


def source_text(s, name):
    text = (Path(__file__).parent / "runtime" / TEMPLATES[name]).read_text()
    if name == "pe.csl":
        text = text.replace(
            "@OBSERVED_PARAMETER@",
            (
                ".observer=observe,.restore_layout=restore_layout,"
                if s["instrumentation"] == "sampled"
                or s["result_layout"] == "transposed_pencils"
                else ""
            ),
        )
    return text


def generate(s, dest):
    for name in TEMPLATES:
        (Path(dest) / name).write_text(source_text(s, name))
