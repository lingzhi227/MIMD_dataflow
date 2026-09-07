"""Reject coherent final/round errors and wrong cyclic ownership in frozen audits."""

import copy, datetime, hashlib, json, shutil, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(src / "implementation"))
from integrity import verify_bundle
from mesh_gemm_sdk import audit

m = verify_bundle(src)
original = json.loads((src / "results.json").read_text())
s = json.loads((src / "schedule.json").read_text())
p = s["P"]
assert s["profile"] == "mesh_cannon.v1"


def coherent_final(r, e, value):
    r["cases"][e]["result"][0] = value
    r["diagnostics"][e]["history"][0][0][-1][0] = value


mutations = {
    "coherent_wrong_final": lambda r: coherent_final(r, 0, 99.0),
    "zero_epoch_stale_result": lambda r: coherent_final(r, 2, 1.0),
    "wrong_partial_round": lambda r: r["diagnostics"][0]["history"][0][0][
        0
    ].__setitem__(0, 99.0),
    "wrong_A_block": lambda r: r["diagnostics"][0]["witness"][0][1][0].__setitem__(
        0, 99.0
    ),
    "wrong_B_block": lambda r: r["diagnostics"][0]["witness"][1][0][1].__setitem__(
        2, 99.0
    ),
    "missing_A_shift": lambda r: r["diagnostics"][0]["progress"][1][1].__setitem__(
        1, p - 2
    ),
    "missing_B_shift": lambda r: r["diagnostics"][0]["progress"][1][1].__setitem__(
        2, p - 2
    ),
    "wrong_invocation": lambda r: r["diagnostics"][1]["progress"][1][1].__setitem__(
        3, 1
    ),
    "undrained_queue": lambda r: r["diagnostics"][0]["queue_last"][1][1].__setitem__(
        0, 0
    ),
    "bad_total_timestamp": lambda r: r["diagnostics"][0]["total_timing"][1][
        1
    ].__setitem__(0, 65536),
    "missing_host_launch": lambda r: r["launches"].pop(),
    "wrong_runtime_instances": lambda r: r.__setitem__("runtime_instances", 2),
}
rows = []
with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "bundle"
    shutil.copytree(src, root, ignore=shutil.ignore_patterns("out", "simfab_traces"))
    audit(root, m)
    for label, mutate in mutations.items():
        r = copy.deepcopy(original)
        mutate(r)
        (root / "results.json").write_text(json.dumps(r))
        try:
            audit(root, m)
        except (AssertionError, ValueError) as e:
            rows.append(dict(case=label, rejected=True, diagnostic=str(e)[:300]))
        else:
            raise AssertionError("Mutation accepted: " + label)
name = "cannon-audit-mutations-" + datetime.datetime.now(
    datetime.timezone.utc
).strftime("%Y%m%dT%H%M%S%fZ")
out = ROOT / "evidence" / name
out.mkdir()
shutil.copyfile(__file__, out / "driver.py")
report = dict(
    passed=True,
    source_bundle=str(src),
    frozen_auditor=str(src / "implementation"),
    source_results_sha256=hashlib.sha256(
        (src / "results.json").read_bytes()
    ).hexdigest(),
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    rejected=rows,
)
out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
print(out.with_suffix(".json"))
