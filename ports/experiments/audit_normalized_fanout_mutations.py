"""Reject coherent branch, warm-state, prefix and live-owner corruptions."""

import copy, datetime, json, shutil, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(src / "implementation"))
from mesh_normalized_fanout_sdk import audit, outputs
from hashlib import sha256

s = json.loads((src / "schedule.json").read_text())
r = json.loads((src / "results.json").read_text())
assert audit(src)["passed"]
p, b, L = s["P"], s["projections"], s["Mt"] * s["Nt"]


def flip(v, key, offset=0, epoch=0):
    v["diagnostics"][epoch][key][0][0][offset] ^= 1


def coherent(v, branch, epoch=0):
    flip(v, "result", branch * L, epoch)
    v["cases"][epoch] = outputs(s, v["diagnostics"][epoch]["result"])


mutations = {f"coherent_branch_{i}": lambda v, i=i: coherent(v, i) for i in range(b)}
mutations.update(
    changed_weights_warm_regression=lambda v: coherent(v, b - 1, 5),
    zero_input_regression=lambda v: coherent(v, b - 1, 2),
    immutable_input_corruption=lambda v: flip(v, "X"),
    immutable_rms_weight_corruption=lambda v: flip(v, "W"),
    stale_epoch=lambda v: v["diagnostics"][1]["progress"][0][0].__setitem__(3, 1),
    missing_branch=lambda v: v["diagnostics"][0]["progress"][0][0].__setitem__(
        2, b - 1
    ),
    missing_round=lambda v: v["diagnostics"][0]["progress"][0][0].__setitem__(
        1, b * p - 1
    ),
    undrained_queue=lambda v: v["diagnostics"][0]["queues"][0][0].__setitem__(0, 0),
    missing_call=lambda v: v["launches"].pop(),
    bad_timestamp=lambda v: v["diagnostics"][0]["timing"][0][0].__setitem__(
        slice(3, 6), v["diagnostics"][0]["timing"][0][0][:3]
    ),
)
for i in range(b):
    mutations[f"wrong_initial_weight_{i}"] = lambda v, i=i: flip(
        v, "projection_input", i * s["Nt"] * s["Nt"]
    )
if s["instrumentation"] == "sampled":
    mutations["wrong_normalization"] = lambda v: flip(v, "normalized")
    for i in range(b):
        mutations[f"wrong_live_owner_branch_{i}"] = lambda v, i=i: flip(
            v, "reuse", i * L
        )
        for j in range(p):
            mutations[f"wrong_prefix_branch_{i}_round_{j}"] = lambda v, i=i, j=j: flip(
                v, "history", (i * p + j) * L
            )
else:
    for key in ("history", "normalized", "reuse"):
        mutations["unexpected_" + key] = lambda v, key=key: flip(v, key)
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
    / "evidence"
    / (
        "normalized-fanout-audit-mutations-"
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
