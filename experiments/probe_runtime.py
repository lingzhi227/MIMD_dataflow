"""Shared frozen execution boundary for CSL-only semantic probes."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import hashlib
import json
import os
from pathlib import Path
import sys

SIF = Path(
    "/home/lingzhi/cerebras/sdk/2.10.1/sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif"
)
SDK_HASH = "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    for name, digest in read(root / "provenance.json")["files"].items():
        assert sha(root / name) == digest, name


def execute(root, timeout=600):
    verify(root)
    assert not (root / "sdk.log").exists(), "Fresh execution required"
    sys.path.insert(0, str(root))
    from sdk_process import run_sdk

    report = dict(
        success=False,
        scope="CSL semantic probe; no HLS application or performance qualification implied",
    )
    try:
        with SIF.open("rb") as stream:
            report["sdk_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        assert report["sdk_sha256"] == SDK_HASH
        with (root / "sdk.log").open("w") as log:
            run_sdk(
                [
                    "/home/lingzhi/cerebras/sdk/2.10.1/cs_python",
                    str(root / "driver.py"),
                    "--worker",
                    str(root),
                ],
                root,
                dict(
                    os.environ,
                    SINGULARITYENV_CS_TARGET="SDR",
                    SINGULARITYENV_PYTHONUNBUFFERED="1",
                ),
                log,
                timeout,
            )
        assert read(root / "results.json")["success"]
        report.update(success=True, results_sha256=sha(root / "results.json"))
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        (root / "execution.json").write_text(json.dumps(report, indent=2) + "\n")
    print(root, flush=True)


def half_vector_worker(root):
    """Single-PE half-vector probes with a declared port schema and warm counter."""
    import subprocess
    import numpy as np
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime,
        MemcpyDataType,
        MemcpyOrder,
        SimfabConfig,
        SdkTarget,
        get_platform,
    )
    from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view

    verify(root)
    os.chdir(root)
    schema = read(root / "schema.json")
    names = [schema["input"], *schema["outputs"], "progress"]
    assert len(set(names)) == len(names)
    subprocess.run(read(root / "sdk-command.json"), check=True)
    runner = SdkRuntime(
        "out",
        get_platform(
            None,
            SimfabConfig(suppress_trace=True, num_threads=4, dump_core=True),
            SdkTarget.WSE3,
        ),
    )
    ids = {name: runner.get_id(name) for name in names}
    runner.load()
    runner.run()
    result = dict(success=False, runtime_instances=1, cases=[])
    try:
        for epoch, values in enumerate(read(root / "inputs.json")):
            x = np.asarray(values, np.float16)
            assert x.shape == (schema["length"],) and np.all(np.isfinite(x))
            runner.memcpy_h2d(
                ids[schema["input"]],
                input_array_to_u32(x, 1, 1),
                0,
                0,
                1,
                1,
                len(x),
                streaming=False,
                data_type=MemcpyDataType.MEMCPY_16BIT,
                order=MemcpyOrder.ROW_MAJOR,
                nonblock=False,
            )
            runner.launch("main", nonblock=False)
            outputs = {}
            for name in names:
                raw = np.zeros(1 if name == "progress" else len(x), np.uint32)
                runner.memcpy_d2h(
                    raw,
                    ids[name],
                    0,
                    0,
                    1,
                    1,
                    len(raw),
                    streaming=False,
                    data_type=MemcpyDataType.MEMCPY_16BIT,
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
                outputs[name] = raw.astype(np.uint16).tolist()
                if name != "progress":
                    np.testing.assert_array_equal(
                        memcpy_view(raw, np.dtype(np.float16)).view(np.uint16),
                        raw.astype(np.uint16),
                    )
            np.testing.assert_array_equal(outputs[schema["input"]], x.view(np.uint16))
            assert outputs["progress"] == [epoch + 1]
            result["cases"].append(outputs)
            (root / "results.json").write_text(json.dumps(result) + "\n")
            print("HALF VECTOR", epoch + 1, flush=True)
    finally:
        runner.stop()
    result["success"] = True
    (root / "results.json").write_text(json.dumps(result) + "\n")


def mesh_half_worker(root):
    """Schema-driven multi-PE half input/raw-word output probes with frozen options.

    Schema declares rows/cols, per-PE extents, launch, optional initialization,
    and immutable input names. It performs no application arithmetic.
    """
    import subprocess, time
    import numpy as np
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime,
        MemcpyDataType,
        MemcpyOrder,
        SimfabConfig,
        SdkTarget,
        get_platform,
    )
    from cerebras.sdk.sdk_utils import input_array_to_u32

    verify(root)
    os.chdir(root)
    schema = read(root / "schema.json")
    rows, cols = schema["rows"], schema["cols"]
    assert 1 <= rows <= 16 and 1 <= cols <= 16
    inputs = schema["inputs"]
    outputs = schema["outputs"]
    word_bits = schema.get("output_word_bits", {})
    assert all(k in outputs and v in (16, 32) for k, v in word_bits.items())
    assert all(
        type(n) is int and n > 0 for n in list(inputs.values()) + list(outputs.values())
    )
    assert all(k in inputs and k in outputs for k in schema.get("immutable", []))

    def stage(operation, epoch=None, port=None):
        tmp = root / "runtime-stage.json.tmp"
        tmp.write_text(
            json.dumps(dict(operation=operation, epoch=epoch, port=port)) + "\n"
        )
        tmp.replace(root / "runtime-stage.json")

    stage("compiling")
    subprocess.run(read(root / "sdk-command.json"), check=True)
    runner = SdkRuntime(
        "out",
        get_platform(
            None, SimfabConfig(**read(root / "runtime-options.json")), SdkTarget.WSE3
        ),
    )
    ids = {k: runner.get_id(k) for k in set(inputs) | set(outputs)}
    stage("loading")
    runner.load()
    stage("starting")
    runner.run()
    result = dict(
        success=False,
        runtime_instances=1,
        cases=[],
        host_call_seconds=[],
        host_timing_scope="Host elapsed for input/launch/output transfers; excludes compilation/load; not device throughput",
    )

    def save():
        tmp = root / "results.json.tmp"
        tmp.write_text(json.dumps(result) + "\n")
        tmp.replace(root / "results.json")

    try:
        if schema.get("initialize"):
            stage("initialize", port=schema["initialize"])
            runner.launch(schema["initialize"], nonblock=False)
        for epoch, batch in enumerate(read(root / "inputs.json")):
            call_started = time.monotonic()
            assert set(batch) == set(inputs)
            for name, n in inputs.items():
                values = np.asarray(batch[name], np.float16)
                assert values.shape == (rows, cols, n) and np.all(np.isfinite(values))
                stage("host_to_device", epoch, name)
                runner.memcpy_h2d(
                    ids[name],
                    input_array_to_u32(values.ravel(), 1, 1),
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
            stage("launch", epoch, schema["launch"])
            runner.launch(schema["launch"], nonblock=False)
            case = {}
            for name, n in outputs.items():
                raw = np.zeros(rows * cols * n, np.uint32)
                stage("device_to_host", epoch, name)
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
                        if word_bits.get(name, 16) == 32
                        else MemcpyDataType.MEMCPY_16BIT
                    ),
                    order=MemcpyOrder.ROW_MAJOR,
                    nonblock=False,
                )
                case[name] = (
                    raw.astype(
                        np.uint32 if word_bits.get(name, 16) == 32 else np.uint16
                    )
                    .reshape(rows, cols, n)
                    .tolist()
                )
            result["cases"].append(case)
            result["host_call_seconds"].append(time.monotonic() - call_started)
            save()
            for name in schema.get("immutable", []):
                np.testing.assert_array_equal(
                    case[name], np.asarray(batch[name], np.float16).view(np.uint16)
                )
            if schema.get("progress"):
                np.testing.assert_array_equal(case[schema["progress"]], epoch + 1)
            print("MESH HALF PROBE", epoch + 1, flush=True)
    finally:
        stage("stopping")
        runner.stop()
    stage("completed")
    result["success"] = True
    save()
