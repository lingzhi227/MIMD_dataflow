"""Reject corruptions of actual FFN snapshots, public operands and protocol evidence."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, datetime, json, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, sha

ROOT = repository_root(__file__)
src = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(src / "implementation"))
from mesh_attention_tail_sdk import audit, audit_cases
from mesh_common import unpack_tiles

assert audit(src)["passed"]
s, m, bs, r = [
    read(src / name)
    for name in ("schedule.json", "semantic.json", "batches.json", "results.json")
]


def flip(v, key, offset=0, epoch=0):
    v["diagnostics"][epoch][key][0][0][offset] ^= 1


def coherent(v, epoch):
    flip(v, "result", epoch=epoch)
    raw = np.asarray(v["diagnostics"][epoch]["result"], np.uint16).view(np.float16)
    v["cases"][epoch] = {
        m["nodes"][-1]["host"]: unpack_tiles(raw, s["Mt"], s["Nt"], "F")
        .astype(float)
        .ravel()
        .tolist()
    }


mutations = dict(
    missing_launch=lambda v: v["launches"].pop(),
    multiple_runtimes=lambda v: v.update(runtime_instances=2),
    undrained_queue=lambda v: v["diagnostics"][0]["queues"][0][0].__setitem__(0, 0),
    zero_interval=lambda v: v["diagnostics"][0]["timing"][0][0].__setitem__(
        slice(3, 6), v["diagnostics"][0]["timing"][0][0][:3]
    ),
    extra_port=lambda v: v["diagnostics"][0].update(unexpected=[0]),
    invalid_half_range=lambda v: v["diagnostics"][0]["normalized"][0][0].__setitem__(
        0, 65536
    ),
    invalid_wide_range=lambda v: v["diagnostics"][0]["wide_accumulator"][0][
        0
    ].__setitem__(0, 2**32),
)
for key in (
    "x",
    "attention_k",
    "attention_v",
    "attention_logits",
    "attention_probability",
    "attention_snapshot",
    "attention_exponents",
    "attention_softmax_history",
    "score_owners",
    "attention_value_left",
    "attention_value_right",
    "output_weight",
    "residual",
    "projection_snapshot",
    "post_projection_z",
    "projection_left_first",
    "projection_right_first",
    "gamma",
    "up_weight",
    "gate_weight",
    "down_weight",
    "normalized",
    "down_snapshot",
    "wide_accumulator",
    "up_accumulator",
    "gate_accumulator",
):
    if key in r["diagnostics"][0]:
        mutations[key] = lambda v, key=key: flip(v, key)
for key, length in (
    ("progress", 8),
    ("rms_progress", 2),
    ("prelude_progress", 4),
    ("score_roots", s["P"]),
    ("score_progress", 3),
    ("attention_progress", 4),
    ("attention_softmax_progress", 6),
):
    for offset in range(length):
        mutations[f"{key}_{offset}"] = lambda v, key=key, offset=offset: flip(
            v, key, offset, 1
        )
for key, length in (
    ("score_history", s["score_length"]),
    ("attention_value_history", s["length"]),
    ("projection_history", s["length"]),
    ("up_history", s["hidden_length"]),
    ("gate_history", s["hidden_length"]),
    ("down_history", s["length"]),
):
    for step in range(s["P"] if s["instrumentation"] == "sampled" else 1):
        mutations[f"{key}_{step}"] = lambda v, key=key, step=step, length=length: flip(
            v, key, step * length
        )
for epoch in range(len(bs)):
    mutations[f"coherent_output_{epoch}"] = lambda v, epoch=epoch: coherent(v, epoch)
for key in (
    "normalized",
    "down_snapshot",
    "up_accumulator",
    "gate_accumulator",
    "wide_accumulator",
):
    if key in r["diagnostics"][0]:
        mutations[f"cancellation_{key}"] = lambda v, key=key: flip(v, key, epoch=6)
# Cover every remaining numerical observer. Timing and queue low bits are
# measured/system state, not fixed numerical words; dedicated protocol
# mutations above already exercise their declared validity constraints.
for key in r["diagnostics"][0]:
    if key not in ("timing", "queues") and not any(
        name == key or name.startswith(key + "_") for name in mutations
    ):
        mutations["remaining_port_" + key] = lambda v, key=key: flip(v, key)
rows = []
for name, mutate in mutations.items():
    v = copy.deepcopy(r)
    mutate(v)
    try:
        audit_cases(s, m, bs, v)
    except (AssertionError, ValueError, KeyError) as e:
        rows.append(dict(name=name, rejected=True, error=str(e)))
    else:
        raise AssertionError("Accepted corruption " + name)
out = (
    ROOT
    / "validation/evidence"
    / (
        "attention-tail-audit-mutations-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + ".json"
    )
)
out.write_text(
    json.dumps(
        dict(
            passed=True,
            new_sdk_execution=False,
            source_bundle=str(src),
            results_sha256=sha(src / "results.json"),
            manifest_sha256=sha(src / "manifest.json"),
            driver_sha256=sha(Path(__file__)),
            mutations=rows,
        ),
        indent=2,
    )
    + "\n"
)
print(out)
