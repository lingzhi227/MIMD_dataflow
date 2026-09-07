"""Verify exact device forward-output handoff and original-input reconstruction."""

import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("forward", type=Path)
p.add_argument("inverse", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()


def read(root, name):
    return json.loads((root / name).read_text())


f = read(a.forward, "schedule.json")
i = read(a.inverse, "schedule.json")
assert (
    f["N"] == i["N"]
    and f["transform"]["direction"] == "forward"
    and i["transform"]["direction"] == "inverse"
    and f["transform"]["norm"] == i["transform"]["norm"]
)
for root in (a.forward, a.inverse):
    code = 'import sys,json;from pathlib import Path;p=Path(sys.argv[1]).resolve();sys.path.insert(0,str(p/"implementation"));from validate import audit;print(json.dumps(audit(p)))'
    assert json.loads(
        subprocess.check_output([sys.executable, "-c", code, str(root)], text=True)
    )["passed"]
assert (
    read(a.forward, "qualification.json")["success"]
    and read(a.inverse, "qualification.json")["success"]
)
fr = read(a.forward, "results.json")
ir = read(a.inverse, "results.json")
ib = read(a.inverse, "batches.json")
original = read(a.forward, "batches.json")
assert len(fr["cases"]) == len(ir["cases"]) == len(ib) == len(original)
rows = []
for source, wire, back, b in zip(fr["cases"], ib, ir["cases"], original):
    np.testing.assert_array_equal(
        np.asarray(source[f["output"]], np.float32).view(np.uint32),
        np.asarray(wire[i["input"]], np.float32).view(np.uint32),
        err_msg="device output must pass unchanged to inverse, including signed zero",
    )
    x = np.asarray(b[f["input"]], np.float64)
    y = np.asarray(back[i["output"]], np.float64)
    assert x.shape == y.shape and np.all(np.isfinite(y))
    error = np.linalg.norm(y - x)
    norm = np.linalg.norm(x)
    assert error <= 2e-5 * norm if norm else np.array_equal(x, y)
    rows.append(
        dict(
            relative_l2_error=float(error / norm) if norm else 0,
            max_abs_error=float(np.max(np.abs(x - y))),
        )
    )
a.output.write_text(
    json.dumps(
        dict(
            passed=True,
            forward=str(a.forward),
            inverse=str(a.inverse),
            new_sdk_execution=False,
            unmodified_device_handoff=True,
            results_sha256=[
                hashlib.sha256((r / "results.json").read_bytes()).hexdigest()
                for r in (a.forward, a.inverse)
            ],
            cases=rows,
        ),
        indent=2,
    )
    + "\n"
)
print(a.output)
