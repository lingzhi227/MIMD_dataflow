"""Isolated WSE3 header/u16-body/f32-tail transport; immutable per-case results."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, json, os, subprocess
from pathlib import Path
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime,
    MemcpyDataType,
    MemcpyOrder,
)

os.chdir(Path(__file__).resolve().parent)
p = argparse.ArgumentParser()
p.add_argument("length", type=int)
p.add_argument("offset", type=int)
p.add_argument("delay", type=int)
p.add_argument("exact_length", type=int)
a = p.parse_args()
subprocess.run(
    [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        "--fabric-dims=9,3",
        "--fabric-offsets=4,1",
        f"--params=length:{a.length},offset:{a.offset},delay:{a.delay},exact_length:{a.exact_length}",
        "-o=out",
        "--memcpy",
        "--channels=1",
    ],
    check=True,
)
r = SdkRuntime("out")
rid = r.get_id("received")
sid = r.get_id("sentinel")
r.load()
r.run()
r.launch("main", nonblock=False)
x = np.zeros(64, np.uint32)
y = np.zeros(3, np.float32)
r.memcpy_d2h(
    x,
    rid,
    1,
    0,
    1,
    1,
    64,
    streaming=False,
    data_type=MemcpyDataType.MEMCPY_16BIT,
    order=MemcpyOrder.COL_MAJOR,
    nonblock=False,
)
r.memcpy_d2h(
    y,
    sid,
    1,
    0,
    1,
    1,
    3,
    streaming=False,
    data_type=MemcpyDataType.MEMCPY_32BIT,
    order=MemcpyOrder.COL_MAJOR,
    nonblock=False,
)
r.stop()
expected = np.arange(1000 + a.offset, 1000 + a.offset + a.length, dtype=np.uint32)
passed = bool(
    np.array_equal(x[: a.length], expected) and np.array_equal(y, [1.25, -2.5, 7.75])
)
Path("result.json").write_text(
    json.dumps(
        dict(
            parameters=vars(a), received=x.tolist(), sentinel=y.tolist(), passed=passed
        ),
        indent=2,
    )
)
print("PASS" if passed else "FAIL")
if not passed:
    raise SystemExit(1)
