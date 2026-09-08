"""Corrupt real completed projection/add/RMS evidence and require audit rejection."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, datetime, json, shutil, sys, tempfile
from pathlib import Path
from hashlib import sha256
import numpy as np

ROOT = repository_root(__file__)
src = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(src / "implementation"))
from mesh_projection_residual_rms_sdk import audit
from mesh_common import unpack_tiles

s = json.loads((src / "schedule.json").read_text())
r = json.loads((src / "results.json").read_text())
m = json.loads((src / "semantic.json").read_text())
assert audit(src)["passed"]


def flip(v, key, offset=0, epoch=0):
    v["diagnostics"][epoch][key][0][0][offset] ^= 1


def coherent(v, epoch=0):
    flip(v, "result", epoch=epoch)
    raw = np.asarray(v["diagnostics"][epoch]["result"], np.uint16).view(np.float16)
    v["cases"][epoch] = {
        m["nodes"][-1]["host"]: unpack_tiles(raw, s["Mt"], s["Nt"], "F")
        .astype(float)
        .ravel()
        .tolist()
    }


mutations = dict(
    missing_call=lambda v: v["launches"].pop(),
    undrained_queue=lambda v: v["diagnostics"][0]["queues"][0][0].__setitem__(0, 0),
    bad_timestamp=lambda v: v["diagnostics"][0]["timing"][0][0].__setitem__(
        slice(3, 6), v["diagnostics"][0]["timing"][0][0][:3]
    ),
    extra_port=lambda v: v["diagnostics"][0].update(unexpected=[0]),
    invalid_raw_range=lambda v: v["diagnostics"][0]["inverse"][0][0].__setitem__(
        0, 65536
    ),
)
for epoch in range(8):
    mutations[f"coherent_output_epoch_{epoch}"] = lambda v, epoch=epoch: coherent(
        v, epoch
    )
for key in ("activation", "weight", "residual", "gamma"):
    mutations["immutable_" + key] = lambda v, key=key: flip(v, key)
for offset in range(4):
    mutations[f"progress_{offset}"] = lambda v, offset=offset: flip(
        v, "progress", offset, 1
    )
for key in (
    "sum",
    "left_first",
    "right_first",
    "local_square_sum",
    "reduced_square_sum",
    "inverse",
):
    mutations[key] = lambda v, key=key: flip(v, key)
for step in range(s["P"] if s["instrumentation"] == "sampled" else 1):
    mutations[f"projection_prefix_{step}"] = lambda v, step=step: flip(
        v, "history", step * s["length"]
    )
mutations["inverse_after_cancellation"] = lambda v: flip(v, "inverse", epoch=6)
mutations["inverse_after_reset"] = lambda v: flip(v, "inverse", epoch=7)
rows = []
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    for f in src.iterdir():
        if f.is_file():
            shutil.copyfile(f, root / f.name)
    shutil.copytree(src / "implementation", root / "implementation")
    for name, mutate in mutations.items():
        v = copy.deepcopy(r)
        mutate(v)
        (root / "results.json").write_text(json.dumps(v))
        try:
            audit(root)
        except (AssertionError, ValueError, KeyError) as e:
            rows.append(dict(name=name, rejected=True, error=str(e)))
        else:
            raise AssertionError("Accepted corruption " + name)
path = (
    ROOT
    / "validation/evidence"
    / (
        "projection-residual-rms-audit-mutations-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + ".json"
    )
)
path.write_text(
    json.dumps(
        dict(
            passed=True,
            new_sdk_execution=False,
            source_bundle=str(src),
            results_sha256=sha256((src / "results.json").read_bytes()).hexdigest(),
            mutations=rows,
        ),
        indent=2,
    )
    + "\n"
)
print(path)
