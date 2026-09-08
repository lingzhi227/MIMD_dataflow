"""Coherent numerical and protocol corruption tests for the fixed-power audit."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, datetime, hashlib, json, shutil, sys, tempfile
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from power_audit import audit
from integrity import verify_bundle

src = Path(sys.argv[1]).resolve()
verify_bundle(src, implementation=False)
original = json.loads((src / "results.json").read_text())


def replicas(r, e, key, i, v):
    for row in r["diagnostics"][e][key]:
        for tile in row:
            tile[i] = v


def coherent(r, e, port, key, i, v):
    r["cases"][e][port][i] = v
    replicas(r, e, key, i, v)


def changed_vector(r, e):
    r["cases"][e]["vector"][0] = 99.0
    r["diagnostics"][e]["cg_solution"][0][0][0] = 99.0


mutations = {
    "zero_operator_false_completion": lambda r: coherent(
        r, 1, "reason", "cg_reason", 0, 0
    ),
    "zero_step_changed_vector": lambda r: changed_vector(r, 3),
    "coherent_wrong_vector": lambda r: changed_vector(r, 0),
    "tiny_norm_lost": lambda r: coherent(r, 6, "norms", "cg_history", 0, 0.0),
    "wrong_reciprocal": lambda r: replicas(r, 0, "power_inverse", 0, 99.0),
    "unused_history": lambda r: coherent(r, 0, "norms", "cg_history", 31, 1.0),
    "history_padding": lambda r: replicas(r, 0, "cg_history", 32, 1.0),
    "unused_inverse": lambda r: replicas(r, 0, "power_inverse", 31, 1.0),
    "wrong_attempts": lambda r: replicas(r, 0, "power_attempts", 0, 15),
    "missing_reduction": lambda r: replicas(r, 0, "cg_progress", 2, 0),
    "undrained_queue": lambda r: r["diagnostics"][0]["cg_queue_last"][1][1].__setitem__(
        0, 0
    ),
    "wrong_operator_witness": lambda r: r["diagnostics"][0]["final_ax"][1][
        1
    ].__setitem__(0, 99.0),
    "wrong_partial_witness": lambda r: r["diagnostics"][0]["partial"][1][1].__setitem__(
        0, 99.0
    ),
    "missing_launch": lambda r: r["launches"].pop(),
}
rows = []
with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "bundle"
    shutil.copytree(src, root, ignore=shutil.ignore_patterns("simfab_traces", "out"))
    audit(root)
    for label, mutate in mutations.items():
        r = copy.deepcopy(original)
        mutate(r)
        (root / "results.json").write_text(json.dumps(r))
        try:
            audit(root)
        except (ValueError, AssertionError) as e:
            rows.append(dict(case=label, rejected=True, diagnostic=str(e)[:300]))
        else:
            raise AssertionError("Mutation accepted: " + label)
    # Inactive operator buffers are intentionally not current-epoch evidence.
    r = copy.deepcopy(original)
    r["diagnostics"][3]["partial"][0][0][0] = 99.0
    r["diagnostics"][3]["final_ax"][0][0][0] = 99.0
    (root / "results.json").write_text(json.dumps(r))
    audit(root)
name = "power-audit-mutations-" + datetime.datetime.now(datetime.timezone.utc).strftime(
    "%Y%m%dT%H%M%S%fZ"
)
out = ROOT / "validation/evidence" / name
shutil.copytree(
    ROOT / "lib", out / "lib", ignore=shutil.ignore_patterns("__pycache__")
)
(out / "experiments").mkdir()
(out / "validation/evidence").mkdir()
shutil.copyfile(__file__, out / "experiments/audit_power_mutations.py")
report = dict(
    passed=True,
    source_bundle=str(src),
    source_results_sha256=hashlib.sha256(
        (src / "results.json").read_bytes()
    ).hexdigest(),
    auditor_snapshot=str(out),
    rejected=rows,
    inactive_zero_step_witness_accepted=True,
)
out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
print(out.with_suffix(".json"))
