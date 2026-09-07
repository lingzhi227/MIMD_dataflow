"""Exercise RMS result audit with coherent corruption; no new device execution."""

import copy, datetime, json, shutil, sys, tempfile, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(src / "implementation"))
from mesh_rms_sdk import audit
from mesh_common import unpack_tiles

s = json.loads((src / "schedule.json").read_text())
m = json.loads((src / "semantic.json").read_text())
original = json.loads((src / "results.json").read_text())
assert audit(src)["passed"]


def corrupt(r, epoch):
    d = r["diagnostics"][epoch]
    d["result"][0][0][0] ^= 1
    r["cases"][epoch][m["nodes"][3]["host"]] = (
        unpack_tiles(
            np.asarray(d["result"], np.uint16).view(np.float16), s["Mt"], s["Nt"], "F"
        )
        .astype(float)
        .ravel()
        .tolist()
    )


mutations = {
    "coherent_result": lambda r: corrupt(r, 0),
    "stale_zero": lambda r: corrupt(r, 3),
    "input_corruption": lambda r: r["diagnostics"][0]["X"][0][0].__setitem__(0, 0),
    "weight_axis": lambda r: r["diagnostics"][0]["W"][0][1].__setitem__(
        0, r["diagnostics"][0]["W"][0][0][0]
    ),
    "missing_phase": lambda r: r["diagnostics"][0]["progress"][0][0].__setitem__(1, 0),
    "stale_epoch": lambda r: r["diagnostics"][1]["progress"][0][0].__setitem__(4, 1),
    "undrained_queue": lambda r: r["diagnostics"][0]["queues"][0][0].__setitem__(0, 0),
    "bad_timestamp": lambda r: r["diagnostics"][0]["timing"][0][0].__setitem__(
        slice(3, 6), r["diagnostics"][0]["timing"][0][0][:3]
    ),
    "missing_call": lambda r: r["launches"].pop(),
    "invalid_half_word": lambda r: r["diagnostics"][0]["result"][0][0].__setitem__(
        0, 65536
    ),
}
if s["instrumentation"] == "sampled":
    for phase in range(3):
        mutations["bad_statistic_" + str(phase)] = lambda r, phase=phase: r[
            "diagnostics"
        ][0]["history"][0][0].__setitem__(phase * s["Mt"], 65535)
rows = []
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    for f in src.iterdir():
        if f.is_file():
            shutil.copyfile(f, root / f.name)
    shutil.copytree(src / "implementation", root / "implementation")
    for name, mutate in mutations.items():
        r = copy.deepcopy(original)
        mutate(r)
        (root / "results.json").write_text(json.dumps(r))
        try:
            audit(root)
        except (ValueError, AssertionError, KeyError) as e:
            rows.append(dict(name=name, rejected=True, error=str(e)))
        else:
            raise AssertionError("Accepted corruption " + name)
p = (
    ROOT
    / "evidence"
    / (
        "rms-audit-mutations-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + ".json"
    )
)
p.write_text(
    json.dumps(
        dict(
            passed=True,
            new_sdk_execution=False,
            source_bundle=str(src),
            results_sha256=hashlib.sha256(
                (src / "results.json").read_bytes()
            ).hexdigest(),
            mutations=rows,
        ),
        indent=2,
    )
    + "\n"
)
print(p)
