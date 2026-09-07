"""Compile and execute isolated task slots with the actual SDK memcpy import.

Failures are evidence, not an inferred global reservation table. Each accepted
slot must increment three times in one runtime. No persistent-context mode.
"""

import datetime, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime,
    MemcpyDataType,
    MemcpyOrder,
)


def worker():
    os.chdir(sys.argv[2])
    runtime = SdkRuntime("out")
    symbol = runtime.get_id("result")
    runtime.load()
    runtime.run()
    actual = []
    for iteration in range(3):
        runtime.launch("main", nonblock=False)
        data = np.zeros(1, np.float32)
        runtime.memcpy_d2h(
            data,
            symbol,
            0,
            0,
            1,
            1,
            1,
            streaming=False,
            data_type=MemcpyDataType.MEMCPY_32BIT,
            order=MemcpyOrder.COL_MAJOR,
            nonblock=False,
        )
        actual.append(float(data[0]))
    runtime.stop()
    Path("runtime.json").write_text(
        json.dumps({"actual": actual, "passed": actual == [1, 2, 3]})
    )


if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    worker()
    raise SystemExit(0)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = (
    ROOT
    / "evidence"
    / (
        "task-slots-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
)
OUT.mkdir()
for name in ("run.py", "layout.csl", "pe.csl"):
    shutil.copy2(HERE / name, OUT / name)
report = {"scope": "WSE3 nonstreaming memcpy, default context only", "cases": []}
(OUT / "source-hashes.json").write_text(
    json.dumps(
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()},
        indent=2,
    )
)
print(OUT, flush=True)
slots = (
    tuple(map(int, sys.argv[1:]))
    if len(sys.argv) > 1
    else (0, 7, 8, 21, 24, 26, 27, 28, 29, 30, 31)
)
for slot in slots:
    d = OUT / str(slot)
    d.mkdir()
    for name in ("layout.csl", "pe.csl"):
        shutil.copy2(OUT / name, d / name)
    os.chdir(d)
    cmd = [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        "--fabric-dims=8,3",
        "--fabric-offsets=4,1",
        "-o=out",
        "--memcpy",
        "--channels=1",
        "--params=slot:" + str(slot),
    ]
    (d / "command.json").write_text(json.dumps(cmd))
    with (d / "compiler.log").open("w") as log:
        compiled = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=90)
    case = {"slot": slot, "compile_exit": compiled.returncode, "runtime_passed": None}
    report["cases"].append(case)
    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    if compiled.returncode:
        print(slot, "compiler_rejected", flush=True)
        continue
    with (d / "runtime.log").open("w") as log:
        executed = subprocess.run(
            [sys.executable, str(OUT / "run.py"), "--worker", str(d)],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=90,
        )
    case["runtime_exit"] = executed.returncode
    if executed.returncode:
        case["runtime_passed"] = False
        (OUT / "results.json").write_text(json.dumps(report, indent=2))
        print(slot, "runtime_failed", flush=True)
        continue
    actual = json.loads((d / "runtime.json").read_text())["actual"]
    case.update(actual=actual, runtime_passed=actual == [1, 2, 3])
    (OUT / "results.json").write_text(json.dumps(report, indent=2))
    print(slot, case["runtime_passed"], flush=True)
    assert case["runtime_passed"], case
