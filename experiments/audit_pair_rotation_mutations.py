"""Reject corrupt gated inputs, activation, product and completion evidence."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, datetime, hashlib, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
src = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(src / "implementation"))
from mesh_pair_rotation_sdk import audit
from mesh_common import unpack_tiles

s = json.loads((src / "schedule.json").read_text())
m = json.loads((src / "semantic.json").read_text())
r = json.loads((src / "results.json").read_text())
assert audit(src)["passed"]


def coherent(v, epoch):
    d = v["diagnostics"][epoch]
    d["result"][0][0][0] ^= 1
    v["cases"][epoch][m["nodes"][-1]["host"]] = (
        unpack_tiles(
            np.asarray(d["result"], np.uint16).view(np.float16), s["Mt"], s["Nt"], "F"
        )
        .astype(float)
        .ravel()
        .tolist()
    )


mutations = {
    "coherent_product": lambda v: coherent(v, 0),
    "changed_gate_regression": lambda v: coherent(v, 1),
    "zero_input_regression": lambda v: coherent(v, 2),
    "input_corruption": lambda v: v["diagnostics"][0]["x"][0][0].__setitem__(0, 0),
    "missing_phase": lambda v: v["diagnostics"][0]["progress"][0][0].__setitem__(1, 0),
    "stale_epoch": lambda v: v["diagnostics"][1]["progress"][0][0].__setitem__(2, 1),
    "bad_timestamp": lambda v: v["diagnostics"][0]["timing"][0][0].__setitem__(
        slice(3, 6), v["diagnostics"][0]["timing"][0][0][:3]
    ),
    "missing_call": lambda v: v["launches"].pop(),
    "invalid_word": lambda v: v["diagnostics"][0]["result"][0][0].__setitem__(0, 65536),
}
mutations["wrong_cosine_input"] = lambda v: v["diagnostics"][0]["cosine"][0][
    0
].__setitem__(0, 65535)
if s["instrumentation"] == "sampled":
    mutations["wrong_product_history"] = lambda v: v["diagnostics"][0]["history"][0][
        0
    ].__setitem__(0, 65535)
else:
    mutations["unexpected_observation"] = lambda v: v["diagnostics"][0]["history"][0][
        0
    ].__setitem__(0, 1)

mutations["wrong_sine_input"] = lambda v: v["diagnostics"][0]["sine"][0][0].__setitem__(
    0, 65535
)

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
        except (ValueError, AssertionError, KeyError) as e:
            rows.append(dict(name=name, rejected=True, error=str(e)))
        else:
            raise AssertionError("Accepted corruption " + name)
p = (
    ROOT
    / "validation/evidence"
    / (
        "pair-rotation-audit-mutations-"
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
