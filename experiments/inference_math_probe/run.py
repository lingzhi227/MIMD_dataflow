"""Prepare separately from executing a frozen SDK half math/source-policy probe."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = repository_root(__file__)
SIF = Path(
    "/home/lingzhi/cerebras/sdk/2.10.1/sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif"
)
SDK_HASH = "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    import numpy as np

    source = ROOT / "third_party/sources/waferllm/Decode/src/decode.csl"
    match = re.search(r"fn fast_exp\(x: f16\) f16 \{.*?\n\}", source.read_text(), re.S)
    assert match is not None
    folder = Path(__file__).parent
    root = (
        ROOT
        / "validation/evidence"
        / (
            "inference-math-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    shutil.copyfile(folder / "layout.csl", root / "layout.csl")
    template = (folder / "pe.csl.in").read_text()
    assert template.count("@SOURCE_FAST_EXP@") == 1
    (root / "pe.csl").write_text(template.replace("@SOURCE_FAST_EXP@", match.group()))
    x = np.asarray(
        [
            -256,
            -32,
            -16,
            -8,
            -4,
            -2,
            -1,
            -0.5,
            -0.0,
            0.0,
            2**-24,
            2**-14,
            0.5,
            1,
            2,
            4,
            8,
            10,
        ],
        np.float16,
    )
    batches = [x.tolist(), x[::-1].tolist(), np.zeros_like(x).tolist()]
    (root / "inputs.json").write_text(json.dumps(batches) + "\n")
    (root / "sdk-command.json").write_text(
        json.dumps(
            [
                "cslc",
                "layout.csl",
                "--arch=wse3",
                "--fabric-dims=8,3",
                "--fabric-offsets=4,1",
                f"--params=N:{len(x)}",
                "-o=out",
                "--memcpy",
                "--channels=1",
            ]
        )
        + "\n"
    )
    shutil.copyfile(__file__, root / "driver.py")
    shutil.copyfile(ROOT / "lib/Runtime/sdk_process.py", root / "sdk_process.py")
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                prepared_only=True,
                source_path=str(source.relative_to(ROOT)),
                source_sha256=sha(source),
                source_function=match.group(),
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                files={p.name: sha(p) for p in root.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT))


def verify(root):
    for name, digest in read(root / "provenance.json")["files"].items():
        assert sha(root / name) == digest, name


def worker(root):
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
    subprocess.run(read(root / "sdk-command.json"), check=True)
    runner = SdkRuntime(
        "out",
        get_platform(
            None,
            SimfabConfig(suppress_trace=True, num_threads=4, dump_core=True),
            SdkTarget.WSE3,
        ),
    )
    ids = {
        name: runner.get_id(name)
        for name in ("x", "sdk_exp", "source_exp", "sdk_sqrt", "progress")
    }
    runner.load()
    runner.run()
    result = dict(success=False, runtime_instances=1, cases=[])
    try:
        for epoch, values in enumerate(read(root / "inputs.json")):
            x = np.asarray(values, np.float16)
            runner.memcpy_h2d(
                ids["x"],
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
            for name in ids:
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
            np.testing.assert_array_equal(outputs["x"], x.view(np.uint16))
            assert outputs["progress"] == [epoch + 1]
            result["cases"].append(outputs)
            (root / "results.json").write_text(json.dumps(result) + "\n")
    finally:
        runner.stop()
    result["success"] = True
    (root / "results.json").write_text(json.dumps(result) + "\n")


def execute(root):
    verify(root)
    assert not (root / "sdk.log").exists(), "Fresh execution required"
    sys.path.insert(0, str(root))
    from sdk_process import run_sdk

    report = dict(
        success=False,
        scope="Primitive SDK/source-policy execution and transport; not full inference or standard-function accuracy qualification",
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
                300,
            )
        assert read(root / "results.json")["success"]
        report["success"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        (root / "execution.json").write_text(json.dumps(report, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--execute", type=Path)
    group.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.execute:
        execute(args.execute.resolve())
    else:
        worker(args.worker.resolve())
