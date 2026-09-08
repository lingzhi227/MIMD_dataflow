"""Fault injection against frozen shared mixed audit using actual SDK results."""

import copy, datetime, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "implementation"))
from integrity import verify_bundle, verify_codegen
from input_attention_mixed_audit import audit_cases
from mesh_input_attention_mixed_sdk import WIDE_PORTS
from mesh_feed_forward_sdk import decode

verify_bundle(root)
verify_codegen(root)
s, m, bs, r = [
    json.loads((root / n).read_text())
    for n in ("schedule.json", "semantic.json", "batches.json", "results.json")
]
assert audit_cases(s, m, bs, r)["passed"]


def flip(v, key, index=0, epoch=0):
    v["diagnostics"][epoch][key][0][0][index] ^= 0x00800000 if key in WIDE_PORTS else 1


def coherent(v):
    flip(v, "result")
    v["cases"][0] = decode(s, m, v["diagnostics"][0])


mutations = dict(
    missing_launch=lambda v: v["launches"].pop(),
    multiple_runtimes=lambda v: v.update(runtime_instances=2),
    incomplete_success=lambda v: v["cases"].pop(),
    extra_port=lambda v: v["diagnostics"][0].update(extra=[0]),
    half_range=lambda v: v["diagnostics"][0]["normalized"][0][0].__setitem__(0, 65536),
    wide_range=lambda v: v["diagnostics"][0]["mixed_v"][0][0].__setitem__(0, 2**32),
    queues=lambda v: v["diagnostics"][0]["queues"][0][0].__setitem__(0, 0),
    zero_interval=lambda v: v["diagnostics"][0]["timing"][0][0].__setitem__(
        slice(3, 6), v["diagnostics"][0]["timing"][0][0][:3]
    ),
    coherent_final=coherent,
)
for key in (
    "residual",
    "gamma",
    "q_weight",
    "k_weight",
    "v_weight",
    "cosine",
    "sine",
    "output_weight",
    "up_weight",
    "gate_weight",
    "down_weight",
    "input_normalized",
    "input_q_raw",
    "input_k_raw",
    "x",
    "attention_k",
    "attention_logits",
    "mixed_v",
    "mixed_probability_snapshot",
    "mixed_a",
    "mixed_projection",
    "mixed_z",
    "mixed_normalized",
    "normalized",
    "down_snapshot",
    "result",
    "wide_accumulator",
    "up_accumulator",
    "gate_accumulator",
):
    mutations[key] = lambda v, key=key: flip(v, key)
for key, count in [
    ("input_prefix_progress", 12),
    ("prelude_progress", 4),
    ("attention_progress", 4),
    ("score_progress", 3),
    ("score_roots", 8),
    ("rms_progress", 2),
    ("attention_softmax_progress", 6),
    ("progress", 8),
]:
    for i in range(count):
        mutations[f"{key}_warm_{i}"] = lambda v, key=key, i=i: flip(v, key, i, 1)
checks = []
for name, mutate in mutations.items():
    value = copy.deepcopy(r)
    mutate(value)
    try:
        audit_cases(s, m, bs, value)
    except (ValueError, AssertionError, KeyError, IndexError) as error:
        checks.append(dict(mutation=name, rejected=True, error=str(error)))
    else:
        raise AssertionError("accepted corruption: " + name)
out = (
    ROOT
    / "evidence"
    / (
        "input-attention-mixed-audit-mutations-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + ".json"
    )
)
report = dict(
    passed=True,
    mutations=len(checks),
    checks=checks,
    source_bundle=str(root),
    results_sha256=hashlib.sha256((root / "results.json").read_bytes()).hexdigest(),
    manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope=__doc__,
)
out.write_text(json.dumps(report, indent=2) + "\n")
print(out, len(checks))
