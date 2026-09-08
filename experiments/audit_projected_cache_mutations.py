"""Fail-closed checks against a completed real SDK PROJECTED CACHE ATTENTION result snapshot."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, copy, hashlib, json, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("bundle", type=Path)
p.add_argument("report", type=Path)
a = p.parse_args()
root = a.bundle.resolve()
assert not a.report.exists()
sys.path.insert(0, str(root / "implementation"))
from integrity import verify_bundle, verify_codegen
from projected_cache_reference import audit_cases

verify_bundle(root)
verify_codegen(root)
read = lambda n: json.loads((root / n).read_text())
s, m, batches, r = [
    read(n) for n in ("schedule.json", "semantic.json", "batches.json", "results.json")
]
assert audit_cases(s, m, batches, r)["passed"] and r["success"] and len(r["cases"]) == 8
mutations = []
for port in sorted(set(r["diagnostics"][0]) - {"progress", "queues", "timing"}):
    for y, x in ((0, 0), (s["P"] - 1, s["P"] - 1)):

        def change(v, port=port, y=y, x=x):
            v["diagnostics"][0][port][y][x][-1] ^= 1

        mutations.append((f"{port}-p{x}_{y}-last-word", change))
for port in ("X", "Q", "K", "V", "W", "probability", "context", "delta", "result"):
    mutations.append(
        (
            port + "-truncated",
            lambda v, port=port: v["diagnostics"][0][port][0][0].pop(),
        )
    )
mutations.extend(
    [
        ("missing-epoch", lambda v: v["cases"].pop()),
        ("false-success", lambda v: v.update(success=False)),
        ("runtime-instances", lambda v: v.update(runtime_instances=2)),
        ("launch-order", lambda v: v["launches"].__setitem__(0, "init_task")),
        ("wrong-output", lambda v: v["cases"][0]["result"].__setitem__(0, 1.0)),
        ("missing-port", lambda v: v["diagnostics"][0].pop("delta")),
        (
            "queue-not-empty",
            lambda v: v["diagnostics"][0]["queues"][0][0].__setitem__(0, 0),
        ),
        (
            "unfinished-phase",
            lambda v: v["diagnostics"][0]["progress"][0][0].__setitem__(5, 0),
        ),
        (
            "epoch-regression",
            lambda v: v["diagnostics"][0]["progress"][0][0].__setitem__(10, 0),
        ),
        (
            "bad-timer-high-word",
            lambda v: v["diagnostics"][0]["timing"][0][0].__setitem__(5, 65535),
        ),
        (
            "zero-cycle",
            lambda v: v["diagnostics"][0]["timing"][0][0].__setitem__(
                slice(3, 6), v["diagnostics"][0]["timing"][0][0][:3]
            ),
        ),
        (
            "word-overflow",
            lambda v: v["diagnostics"][0]["delta"][0][0].__setitem__(0, 65536),
        ),
    ]
)
mutations.extend(
    [
        (
            "last-call-delta-corruption",
            lambda v: v["diagnostics"][-1]["delta"][-1][-1].__setitem__(
                0, v["diagnostics"][-1]["delta"][-1][-1][0] ^ 1
            ),
        ),
        (
            "last-call-immutable-cache",
            lambda v: v["diagnostics"][-1]["K"][-1][-1].__setitem__(
                0, v["diagnostics"][-1]["K"][-1][-1][0] ^ 1
            ),
        ),
        (
            "last-call-stale-observations",
            lambda v: v["diagnostics"].__setitem__(
                -1, copy.deepcopy(v["diagnostics"][-2])
            ),
        ),
        (
            "nonzero-call-stale-output",
            lambda v: v["cases"].__setitem__(1, copy.deepcopy(v["cases"][0])),
        ),
    ]
)
mutations.extend(
    [
        (
            "all-negative-global-max-zero",
            lambda v: v["diagnostics"][3]["maximum"][7][7].__setitem__(0, 0),
        ),
        (
            "nonzero-cache-transposed-local",
            lambda v: v["diagnostics"][4]["K"][7][3].reverse(),
        ),
        (
            "max-padding-zero",
            lambda v: v["diagnostics"][0]["maximum"][7][7].__setitem__(-1, 0),
        ),
        (
            "sum-padding-nonzero",
            lambda v: v["diagnostics"][0]["sums"][7][7].__setitem__(-1, 15360),
        ),
    ]
)
# Public auxiliary outputs must be checked, not only the residual. The final
# K/V replicas remain live even though the supplied old cache is unchanged.
for port in ("new_key", "new_value"):
    mutations.append(
        (
            port + "-decoded-corruption",
            lambda v, port=port: v["cases"][1][port].__setitem__(
                0, v["cases"][1][port][0] + 1
            ),
        )
    )
for port in (
    "projections",
    "rotated_key",
    "query_pair_history",
    "key_pair_history",
    "rms_history",
    "projection_partial",
):
    mutations.append(
        (
            port + "-last-call-stale",
            lambda v, port=port: v["diagnostics"][-1].__setitem__(
                port, copy.deepcopy(v["diagnostics"][-2][port])
            ),
        )
    )
mutations.extend(
    [
        ("success-integer", lambda v: v.update(success=1)),
        ("runtime-instances-boolean", lambda v: v.update(runtime_instances=True)),
        ("missing-diagnostic-epoch", lambda v: v["diagnostics"].pop()),
        ("extra-epoch", lambda v: v["cases"].append(copy.deepcopy(v["cases"][-1]))),
    ]
)
reports = []
for name, mutate in mutations:
    damaged = copy.deepcopy(r)
    mutate(damaged)
    try:
        audit_cases(s, m, batches, damaged)
    except (ValueError, AssertionError) as error:
        reports.append(dict(name=name, rejected=True, error=str(error)[:220]))
    else:
        raise AssertionError("mutation accepted: " + name)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
a.report.write_text(
    json.dumps(
        dict(
            passed=True,
            mutations=len(reports),
            cases=reports,
            source_bundle=str(root),
            results_sha256=sha(root / "results.json"),
            manifest_sha256=sha(root / "manifest.json"),
            driver_sha256=sha(Path(__file__)),
            scope="Actual full SDK snapshot rejected after numeric, replica, ownership, lifecycle and timestamp corruption through its frozen auditor",
        ),
        indent=2,
    )
    + "\n"
)
print("PROJECTED CACHE ATTENTION MUTATIONS PASS", len(reports))
