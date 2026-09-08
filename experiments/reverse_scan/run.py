
from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)

import os, subprocess, json
from pathlib import Path
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime,
    MemcpyDataType,
    MemcpyOrder,
)

os.chdir(Path(__file__).resolve().parent)
subprocess.run(
    [
        "cslc",
        "layout.csl",
        "--arch=wse3",
        "--fabric-dims=8,3",
        "--fabric-offsets=4,1",
        "-o=out",
        "--memcpy",
        "--channels=1",
    ],
    check=True,
)
r = SdkRuntime("out")
idx = r.get_id("result")
r.load()
r.run()
r.launch("main", nonblock=False)
data = np.zeros(8, np.float32)
r.memcpy_d2h(
    data,
    idx,
    0,
    0,
    1,
    1,
    8,
    streaming=False,
    data_type=MemcpyDataType.MEMCPY_32BIT,
    order=MemcpyOrder.COL_MAJOR,
    nonblock=False,
)
r.stop()
expected = [0, 65535, 6, 65535, 65, 0, 71, 65535]
Path("result.json").write_text(
    json.dumps(
        dict(
            actual=data.tolist(),
            expected=expected,
            passed=bool(np.array_equal(data, expected)),
        ),
        indent=2,
    )
)
np.testing.assert_array_equal(data, expected)
print("PASS")
