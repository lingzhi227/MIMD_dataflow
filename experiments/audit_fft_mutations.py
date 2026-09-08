"""Corrupt preserved result copies; never edit executed evidence or rerun a simulator."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import copy, datetime, hashlib, json, shutil, sys, tempfile
from pathlib import Path

ROOT = repository_root(__file__)
src = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(src / "implementation"))
from mesh_fft_sdk import audit
from mesh_fft import unpack, interleave
import mesh_fft

s = json.loads((src / "schedule.json").read_text())
original = json.loads((src / "results.json").read_text())
assert audit(src)["passed"]


def coherent(r, epoch):
    d = r["diagnostics"][epoch]
    d["packed_output"][0][0][0] += 10
    physical = unpack(d["packed_output"], s)
    logical = (
        mesh_fft.logical_output(physical, s)
        if hasattr(mesh_fft, "logical_output")
        else physical
    )
    r["cases"][epoch][s["output"]] = interleave(logical)


mutations = {
    "coherent_numerical_error": lambda r: coherent(r, 0),
    "zero_reset_stale": lambda r: coherent(r, 2),
    "wrong_host_layout": lambda r: r["cases"][0][s["output"]].__setitem__(0, 999),
    "missing_callback": lambda r: r["diagnostics"][0]["progress"][0][0].__setitem__(
        1, 0
    ),
    "stale_epoch": lambda r: r["diagnostics"][1]["progress"][0][0].__setitem__(0, 1),
    "undrained_owned_iq": lambda r: r["diagnostics"][0]["queues"][0][0].__setitem__(
        0, 0
    ),
    "undrained_owned_oq": lambda r: r["diagnostics"][0]["queues"][0][0].__setitem__(
        1, 0
    ),
    "invalid_timestamp": lambda r: r["diagnostics"][0]["timing"][0][0].__setitem__(
        0, 65536
    ),
    "missing_result": lambda r: r["cases"].pop(),
    "new_runtime_per_call": lambda r: r.__setitem__("runtime_instances", 2),
}
if s.get("internal_phases_observed"):
    mutations.update(
        missing_internal_callback=lambda r: r["diagnostics"][0]["phase_counts"][0][
            0
        ].__setitem__(3, 0),
        wrong_transpose_sample=lambda r: r["diagnostics"][0]["phase_samples"][0][
            0
        ].__setitem__(4 * s["T"] ** 2, 999),
    )
    if s.get("phase_count") == 5:
        mutations["unexpected_restoration_callback"] = lambda r: r["diagnostics"][0][
            "phase_counts"
        ][0][0].__setitem__(5, 1)
rows = []
with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "bundle"
    root.mkdir()
    for f in src.iterdir():
        if f.is_file():
            shutil.copyfile(f, root / f.name)
    shutil.copytree(src / "implementation", root / "implementation")
    for name, fn in mutations.items():
        r = copy.deepcopy(original)
        fn(r)
        (root / "results.json").write_text(json.dumps(r))
        try:
            audit(root)
        except (ValueError, AssertionError, KeyError) as e:
            rows.append(dict(name=name, rejected=True, error=str(e)))
        else:
            raise AssertionError("accepted corruption " + name)
p = (
    ROOT
    / "validation/evidence"
    / (
        "fft-audit-mutations-"
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
