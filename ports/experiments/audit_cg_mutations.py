"""Fault-inject copied SDK records to check rejection boundaries of the CG auditor."""

import argparse, copy, datetime, hashlib, json, shutil, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toolchain"))
from mesh_cg_sdk import audit
from integrity import verify_bundle

p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
a = p.parse_args()
src = a.bundle.resolve()
m = verify_bundle(src, implementation=False)
original = json.loads((src / "results.json").read_text())


def coherent(r, epoch, port, diagnostic, index, value):
    r["cases"][epoch][port][index] = value
    for row in r["diagnostics"][epoch][diagnostic]:
        for tile in row:
            tile[index] = value


def coherent_alpha(r):
    for row in r["diagnostics"][0]["cg_scalars"]:
        for tile in row:
            tile[0] = 99.0


mutations = {
    "underflow_false_convergence": lambda r: coherent(
        r, 7, "reason", "cg_reason", 0, 0
    ),
    "underflow_lost_stable_norm": lambda r: coherent(
        r, 7, "true_residual_norm", "cg_true_norm", 0, 0.0
    ),
    "history_tail_corruption": lambda r: coherent(
        r,
        0,
        "residual_squared",
        "cg_history",
        len(r["cases"][0]["residual_squared"]) - 1,
        1.0,
    ),
    "solution_transport_mismatch": lambda r: r["cases"][0]["solution"].__setitem__(
        0, 99.0
    ),
    "nonreplicated_status": lambda r: r["diagnostics"][0]["cg_reason"][1][
        1
    ].__setitem__(0, 1),
    "undrained_queue": lambda r: r["diagnostics"][0]["cg_queue_last"][1][1].__setitem__(
        0, 0
    ),
    "coherent_bad_alpha": coherent_alpha,
    "bad_partial_witness": lambda r: r["diagnostics"][0]["partial"][1][1].__setitem__(
        0, 99.0
    ),
    "missing_host_launch": lambda r: r["launches"].pop(),
}

if "cg_weights" in original["diagnostics"][0]:

    def bad_weight(r):
        for row in r["diagnostics"][0]["cg_weights"]:
            for tile in row:
                tile[0] *= 2

    mutations["coherent_wrong_weighted_inner_product"] = bad_weight
    mutations["wrong_device_diagonal_inverse"] = lambda r: r["diagnostics"][0][
        "cg_diagonal"
    ][1][1].__setitem__(0, 99.0)
if "bi_products" in original["diagnostics"][0]:

    def all_tiles(r, epoch, key, index, value):
        for row in r["diagnostics"][epoch][key]:
            for tile in row:
                tile[index] = value

    mutations["wrong_bicgstab_denominator"] = lambda r: all_tiles(
        r, 0, "bi_products", 0, 99.0
    )
    mutations["wrong_bicgstab_omega"] = lambda r: all_tiles(r, 0, "cg_scalars", 1, 0.0)
    mutations["false_early_s_branch"] = lambda r: all_tiles(r, 0, "bi_early", 0, 0)
    mutations["wrong_zero_omega_failure_stage"] = lambda r: all_tiles(
        r, 9, "bi_failure", 0, 21
    )
    mutations["missing_as_call"] = lambda r: all_tiles(r, 0, "bi_progress", 1, 0)
rows = []
with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "bundle"
    shutil.copytree(src, root)
    audit(root, m)
    for label, mutate in mutations.items():
        r = copy.deepcopy(original)
        mutate(r)
        (root / "results.json").write_text(json.dumps(r))
        try:
            audit(root, m)
        except (ValueError, AssertionError) as e:
            rows.append(
                dict(
                    case=label,
                    rejected=True,
                    diagnostic=(
                        str(e).splitlines()[0]
                        if str(e).splitlines()
                        else type(e).__name__
                    ),
                )
            )
        else:
            raise AssertionError("Mutation accepted: " + label)
out = (
    ROOT
    / "evidence"
    / (
        "cg-audit-mutations-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + ".json"
    )
)
snapshot = out.with_suffix("")
shutil.copytree(
    ROOT / "toolchain",
    snapshot / "toolchain",
    ignore=shutil.ignore_patterns("__pycache__"),
)
(snapshot / "experiments").mkdir()
(snapshot / "evidence").mkdir()
shutil.copyfile(__file__, snapshot / "experiments/audit_cg_mutations.py")
out.write_text(
    json.dumps(
        dict(
            passed=True,
            source_bundle=str(src),
            auditor_snapshot=str(snapshot),
            auditor_sources={
                name: hashlib.sha256(
                    (ROOT / "toolchain" / name).read_bytes()
                ).hexdigest()
                for name in ["mesh_cg_sdk.py", "bicgstab_sdk.py", "solver_audit.py"]
            },
            results_sha256=hashlib.sha256(
                (src / "results.json").read_bytes()
            ).hexdigest(),
            auditor_sha256=hashlib.sha256(
                (ROOT / "toolchain/mesh_cg_sdk.py").read_bytes()
            ).hexdigest(),
            cases=rows,
        ),
        indent=2,
    )
    + "\n"
)
print(out)
