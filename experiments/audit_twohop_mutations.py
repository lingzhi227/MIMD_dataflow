"""Reject coherent final/round errors and wrong cyclic ownership in frozen audits."""

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
from integrity import verify_bundle
from twohop_sdk import audit

m = verify_bundle(src)
original = json.loads((src / "results.json").read_text())
s = json.loads((src / "schedule.json").read_text())
p = s["P"]
assert s["profile"] == "mesh_twohop.v1"


def coherent_final(r, e, value):
    import struct

    word = struct.unpack("<H", struct.pack("<e", value))[0]
    r["cases"][e]["result"][0] = value
    r["diagnostics"][e]["result_bits"][0][0][0] = word
    r["diagnostics"][e]["history_bits"][0][0][-1][0] = word


mutations = {
    "coherent_wrong_final": lambda r: coherent_final(r, 0, 99.0),
    "zero_epoch_stale_result": lambda r: coherent_final(r, 2, 1.0),
    "one_ulp_wrong_prefix": lambda r: r["diagnostics"][0]["history_bits"][0][0][
        0
    ].__setitem__(0, r["diagnostics"][0]["history_bits"][0][0][0][0] ^ 1),
    "wrong_X_block": lambda r: r["diagnostics"][0]["witness_bits"][0][1][0].__setitem__(
        0, 0
    ),
    "wrong_W_block": lambda r: r["diagnostics"][0]["witness_bits"][1][0][1].__setitem__(
        2, 0
    ),
    "missing_X_callback": lambda r: r["diagnostics"][0]["progress"][1][1].__setitem__(
        2, p - 1
    ),
    "missing_W_callback": lambda r: r["diagnostics"][0]["progress"][1][1].__setitem__(
        3, p - 1
    ),
    "missing_alignment": lambda r: r["diagnostics"][0]["progress"][1][1].__setitem__(
        4, 0
    ),
    "wrong_invocation": lambda r: r["diagnostics"][1]["progress"][1][1].__setitem__(
        6, 1
    ),
    "undrained_queue": lambda r: r["diagnostics"][0]["queue"][1][1].__setitem__(0, 0),
    "bad_timestamp_word": lambda r: r["diagnostics"][0]["total_timing"][1][
        1
    ].__setitem__(0, 65536),
    "missing_host_launch": lambda r: r["launches"].pop(),
    "wrong_runtime_instances": lambda r: r.__setitem__("runtime_instances", 2),
}
rows = []
with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "bundle"
    shutil.copytree(src, root, ignore=shutil.ignore_patterns("out", "simfab_traces"))
    audit(root)
    for label, mutate in mutations.items():
        r = copy.deepcopy(original)
        mutate(r)
        (root / "results.json").write_text(json.dumps(r))
        try:
            audit(root)
        except (AssertionError, ValueError) as e:
            rows.append(dict(case=label, rejected=True, diagnostic=str(e)[:300]))
        else:
            raise AssertionError("Mutation accepted: " + label)
name = "twohop-audit-mutations-" + datetime.datetime.now(
    datetime.timezone.utc
).strftime("%Y%m%dT%H%M%S%fZ")
out = ROOT / "validation/evidence" / name
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
