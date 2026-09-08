"""Reject replica, batch-major, source-order and lifetime corruption through frozen audit."""

import argparse, copy, json, shutil, sys, tempfile
from pathlib import Path
from probe_runtime import read, sha

p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
p.add_argument("report", type=Path)
a = p.parse_args()
root = a.bundle.resolve()
assert not a.report.exists()
sys.path.insert(0, str(root / "implementation"))
from mesh_pair_rotation_sdk import audit, unpack
import numpy as np

assert audit(root)["passed"]
r = read(root / "results.json")
s = read(root / "schedule.json")
changes = {}
for name in ("x", "cosine", "sine", "result", "history", "progress"):
    for y, x in ((0, 0), (s["rows"] - 1, s["cols"] - 1)):
        changes[f"{name}_pe{y}_{x}"] = lambda v, k=name, y=y, x=x: v["diagnostics"][-1][
            k
        ][y][x].__setitem__(-1, v["diagnostics"][-1][k][y][x][-1] ^ 1)
for name in ("x", "cosine", "sine", "result", "history", "progress", "timing"):
    changes["truncated_" + name] = lambda v, k=name: v["diagnostics"][0][k][0][0].pop()
changes["stale_epoch"] = lambda v: v["diagnostics"][-1]["progress"][7][7].__setitem__(
    2, 1
)
changes["missing_call"] = lambda v: v["cases"].pop()
changes["missing_diagnostics"] = lambda v: v["diagnostics"].pop()
changes["extra_launch"] = lambda v: v["launches"].append("hls_main")
changes["multiple_runtimes"] = lambda v: v.update(runtime_instances=2)
changes["unsuccessful"] = lambda v: v.update(success=False)
changes["bad_word"] = lambda v: v["diagnostics"][0]["result"][0][0].__setitem__(
    0, 65536
)
changes["zero_time"] = lambda v: v["diagnostics"][0]["timing"][0][0].__setitem__(
    slice(3, 6), v["diagnostics"][0]["timing"][0][0][:3]
)


def coherent(v):
    d = v["diagnostics"][0]
    d["result"][0][0][0] ^= 1
    v["cases"][0] = {
        "rotated": unpack(s, np.asarray(d["result"], np.uint16).view(np.float16))
        .ravel()
        .tolist()
    }


changes["coherent_host_and_device"] = coherent
changes["swapped_feature_axes"] = lambda v: v["diagnostics"][2].update(
    result=np.asarray(v["diagnostics"][2]["result"]).transpose(1, 0, 2).tolist()
)
changes["stale_products"] = lambda v: v["diagnostics"][-1].update(
    history=copy.deepcopy(v["diagnostics"][0]["history"])
)
reports = []
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    for f in root.iterdir():
        if f.is_file():
            shutil.copyfile(f, tmp / f.name)
    shutil.copytree(root / "implementation", tmp / "implementation")
    for name, mutate in changes.items():
        damaged = copy.deepcopy(r)
        mutate(damaged)
        (tmp / "results.json").write_text(json.dumps(damaged))
        try:
            audit(tmp)
        except (AssertionError, ValueError, KeyError) as error:
            reports.append(dict(name=name, rejected=True, error=str(error)[:220]))
        else:
            raise AssertionError("Accepted mutation " + name)
a.report.write_text(
    json.dumps(
        dict(
            passed=True,
            source_bundle=str(root),
            mutations=len(reports),
            checks=reports,
            results_sha256=sha(root / "results.json"),
            manifest_sha256=sha(root / "manifest.json"),
            driver_sha256=sha(Path(__file__)),
        ),
        indent=2,
    )
    + "\n"
)
print("BATCHED PAIR MUTATIONS PASS", len(reports))
