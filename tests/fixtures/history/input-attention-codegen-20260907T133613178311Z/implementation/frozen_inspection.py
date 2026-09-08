"""Audit one immutable result snapshot using its bundle's frozen implementation."""

import json, subprocess, sys
from pathlib import Path


def completed(root, results):
    code = """
import importlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/"implementation"))
from integrity import verify_bundle,verify_codegen
verify_bundle(root)
verify_codegen(root)
s=json.loads((root/"schedule.json").read_text())
m=json.loads((root/"semantic.json").read_text())
profile=s["profile"]
allowed={"mesh_mlp.v1":"mesh_mlp", "mesh_projection_residual_rms.v1":"mesh_projection_residual_rms", "mesh_feed_forward.v1":"mesh_feed_forward", "mesh_prefill_tail.v1":"mesh_prefill_tail", "mesh_attention_tail.v1":"mesh_attention_tail"}
base=allowed[profile]
assert importlib.import_module(base).plan(m)==s,"frozen schedule regeneration mismatch"
audit=importlib.import_module(base+"_sdk").audit_cases
r=audit(s,m,json.loads((root/"batches.json").read_text()),json.load(sys.stdin),require_complete=False)
print(json.dumps(r))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(root).resolve())],
        input=json.dumps(results),
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise ValueError(
            "frozen completed-call inspection failed: " + result.stderr.strip()
        )
    return json.loads(result.stdout)
