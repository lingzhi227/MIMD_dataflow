"""Actual frozen SDK data: reject corrupted replicas, padding, inputs and lifecycle."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, datetime, hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
ROOT = repository_root(__file__)
sys.path.insert(0, str(root / "implementation"))
from integrity import verify_bundle, verify_codegen
from mesh_batched_fanout_sdk import audit_cases, decode

verify_bundle(root)
verify_codegen(root)
s, m, bs, r = [
    json.loads((root / n).read_text())
    for n in ("schedule.json", "semantic.json", "batches.json", "results.json")
]
assert audit_cases(s, m, bs, r)["passed"]


def flip(v, key, e=0, y=0, x=0, i=0):
    v["diagnostics"][e][key][y][x][i] ^= 1


def coherent(v):
    flip(v, "projections")
    v["cases"][0] = decode(s, m, v["diagnostics"][0])


mutations = dict(
    missing_launch=lambda v: v["launches"].pop(),
    multiple_runtimes=lambda v: v.update(runtime_instances=2),
    missing_case=lambda v: v["cases"].pop(),
    incomplete=lambda v: v.update(success=False),
    extra_port=lambda v: v["diagnostics"][0].update(extra=[0]),
    coherent_result=coherent,
    invalid_word=lambda v: v["diagnostics"][0]["X"][0][0].__setitem__(0, 65536),
    queue=lambda v: v["diagnostics"][0]["queues"][0][0].__setitem__(0, 0),
    timing=lambda v: v["diagnostics"][0]["timing"][0][0].__setitem__(
        slice(3, 6), v["diagnostics"][0]["timing"][0][0][:3]
    ),
)
for key in ("X", "W", "result", "sums", "history", "weights", "partial", "projections"):
    for point in ((0, 0, 0), (7, 7, 7)):
        e, y, x = point
        mutations[f"{key}_{e}_{y}_{x}"] = lambda v, key=key, e=e, y=y, x=x: flip(
            v, key, e, y, x
        )
for i in range(8):
    mutations[f"warm_progress_{i}"] = lambda v, i=i: flip(v, "progress", e=1, i=i)
for key, offset in (
    ("sums", s["B"]),
    ("history", s["B"]),
    ("history", s["padded_batches"] + s["B"]),
):
    mutations[f"padding_{key}_{offset}"] = lambda v, key=key, i=offset: flip(
        v, key, e=1, y=7, x=7, i=i
    )
for binding in s["branch_bindings"]:
    for key in ("partial", "projections"):
        for edge in (0, binding["length"] - 1):
            index = binding["offset"] + edge
            mutations[f"branch{binding['index']}_{key}_{edge}"] = (
                lambda v, key=key, index=index: flip(v, key, e=7, y=7, x=7, i=index)
            )
checks = []
for name, mutate in mutations.items():
    value = copy.deepcopy(r)
    mutate(value)
    try:
        audit_cases(s, m, bs, value)
    except (ValueError, AssertionError, KeyError, IndexError) as error:
        checks.append(dict(mutation=name, rejected=True, error=str(error)))
    else:
        raise AssertionError("accepted corruption " + name)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
out = (
    ROOT
    / "validation/evidence"
    / (
        "batched-fanout-audit-mutations-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + ".json"
    )
)
assert not out.exists()
out.write_text(
    json.dumps(
        dict(
            passed=True,
            mutations=len(checks),
            checks=checks,
            source_bundle=str(root),
            results_sha256=sha(root / "results.json"),
            manifest_sha256=sha(root / "manifest.json"),
            driver_sha256=sha(Path(__file__)),
        ),
        indent=2,
    )
    + "\n"
)
print(out, len(checks))
