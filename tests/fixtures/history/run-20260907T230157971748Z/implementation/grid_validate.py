"""Audit every time step and physical link volume, not only final field."""

import json, struct
from pathlib import Path
from frontend import check
from float32 import close
from grid_ir import simulate
from grid_plan import plan
from grid_backend import actor


def audit(root, manifest):
    root = Path(root)
    m = json.loads((root / "semantic.json").read_text())
    s = json.loads((root / "schedule.json").read_text())
    r = json.loads((root / "results.json").read_text())
    batches = json.loads((root / "batches.json").read_text())
    check(s == plan(m), "grid schedule regeneration")
    check(r["success"] and len(r["cases"]) == len(batches), "grid execution incomplete")
    histories = {n["id"]: [] for n in s["nodes"]}
    outname = m["nodes"][3]["host"]
    observations = 0
    for batch, actual in zip(batches, r["cases"]):
        values, h = simulate(m, batch, True)
        check(close(actual, {outname: values}), "grid final mismatch")
        for name, v in h.items():
            histories[name].extend(v)
    for n in s["nodes"]:
        name = n["id"]
        check(
            (root / (name + ".csl")).read_text() == actor(n, s), "grid codegen mismatch"
        )
        d = r["diagnostics"][name]
        actual = [struct.unpack("f", struct.pack("I", v))[0] for v in d["history"]]
        check(close(actual, histories[name]), "grid timestep mismatch " + name)
        observations += len(actual)
        counts = {
            "epochs": s["epochs"],
            "step": s["grid"]["steps"],
            "received": s["epochs"]
            * s["grid"]["steps"]
            * s["wire_z"]
            * len(n["neighbors"]),
            "sent": s["epochs"]
            * s["grid"]["steps"]
            * s["wire_z"]
            * len(n["neighbors"]),
            "host_received": s["epochs"] * n["ingress_size"],
            "host_sent": s["epochs"] * n["egress_size"],
            "forwarded": s["epochs"] * n["forward_size"],
            "collected": s["epochs"] * n["collect_size"],
        }
        for key, value in counts.items():
            check(d[key] == [value], "grid lifecycle " + name + "." + key)
        observations += len(counts)
    report = {
        "passed": True,
        "output_values": manifest["expected_output_values"],
        "internal_observations": observations,
        "epochs": s["epochs"],
        "actors": len(s["nodes"]),
        "steps": s["grid"]["steps"],
        "source_sha256": manifest["source_sha256"],
    }
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
