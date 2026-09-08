"""Reject coherent score, root, live-owner and warm-state corruption."""

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
from mesh_mlp_sdk import audit
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
    coherent_output=coherent,
    changed_dense_warm=lambda v: coherent(v, 1),
    zero_down_warm=lambda v: coherent(v, 4),
    missing_call=lambda v: v["launches"].pop(),
    undrained_queue=lambda v: v["diagnostics"][0]["queues"][0][0].__setitem__(0, 0),
    bad_timestamp=lambda v: v["diagnostics"][0]["timing"][0][0].__setitem__(
        slice(3, 6), v["diagnostics"][0]["timing"][0][0][:3]
    ),
)
for key in ("x", "up_weight", "gate_weight", "down_weight"):
    mutations["immutable_" + key] = lambda v, key=key: flip(v, key)
for i in range(8):
    mutations[f"progress_{i}"] = lambda v, i=i: flip(v, "progress", i, 1)
for key, length in [
    ("up_history", s["hidden_length"]),
    ("gate_history", s["hidden_length"]),
    ("down_history", s["length"]),
]:
    for step in range(s["P"] if s["instrumentation"] == "sampled" else 1):
        mutations[f"{key}_{step}"] = lambda v, key=key, length=length, step=step: flip(
            v, key, step * length
        )
for key in (
    "gate_snapshot",
    "hidden_snapshot",
    "activated_gate",
    "left_first",
    "right_first",
):
    mutations[key] = lambda v, key=key: flip(v, key)
if s["instrumentation"] == "sampled":
    for phase in (1, 2):
        mutations[f"left_phase_{phase}"] = lambda v, phase=phase: flip(
            v, "left_first", phase * s["length"]
        )
        mutations[f"right_phase_{phase}"] = lambda v, phase=phase: flip(
            v, "right_first", phase * s["weight_length"]
        )
wide_low_bit_offset = None
if s.get("down_accumulation") == "block_f32":
    # A dense sum can land exactly on a half rounding midpoint. Select a real
    # accumulator word whose low-bit corruption leaves the half result intact.
    for offset, word in enumerate(r["diagnostics"][0]["wide_accumulator"][0][0]):
        pair = (
            np.asarray([word, word ^ 1], np.uint32)
            .view(np.float32)
            .astype(np.float16)
            .view(np.uint16)
        )
        if pair[0] == pair[1]:
            wide_low_bit_offset = offset
            break
    assert wide_low_bit_offset is not None, "need a corruption hidden by half rounding"
    mutations["f32_accumulator_low_bit"] = lambda v: flip(
        v, "wide_accumulator", wide_low_bit_offset
    )
    mutations["f32_accumulator_reset"] = lambda v: flip(v, "wide_accumulator", epoch=7)
    mutations["cancellation_output"] = lambda v: coherent(v, 6)
    mutations["post_cancellation_zero"] = lambda v: coherent(v, 7)
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
        "mlp-audit-mutations-"
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
            wide_low_bit_offset=wide_low_bit_offset,
        ),
        indent=2,
    )
    + "\n"
)
print(path)
