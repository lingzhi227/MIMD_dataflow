"""Audit one immutable result snapshot using its bundle's frozen implementation."""

import json, subprocess, sys
from pathlib import Path


def completed(root, results):
    code = """
import importlib,inspect,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/"implementation"))
from integrity import verify_bundle,verify_codegen
verify_bundle(root)
verify_codegen(root)
s=json.loads((root/"schedule.json").read_text())
m=json.loads((root/"semantic.json").read_text())
profile=s["profile"]
allowed={"mesh_cache_attention.v1":"mesh_cache_attention","mesh_batched_feed_forward.v1":"mesh_batched_feed_forward","mesh_batched_fanout.v1":"mesh_batched_fanout","mesh_batched_rms.v1":"mesh_batched_rms", "mesh_mlp.v1":"mesh_mlp", "mesh_projection_residual_rms.v1":"mesh_projection_residual_rms", "mesh_feed_forward.v1":"mesh_feed_forward", "mesh_prefill_tail.v1":"mesh_prefill_tail", "mesh_attention_tail.v1":"mesh_attention_tail", "mesh_input_attention_mixed.v1":"mesh_input_attention_mixed"}
base=allowed[profile]
assert importlib.import_module(base).plan(m)==s,"frozen schedule regeneration mismatch"
audit=importlib.import_module("batched_ffn_reference" if profile=="mesh_batched_feed_forward.v1" else "input_attention_mixed_audit" if profile=="mesh_input_attention_mixed.v1" else base+"_sdk").audit_cases
inputs=json.loads((root/"batches.json").read_text())
results=json.load(sys.stdin)
if profile in ("mesh_batched_feed_forward.v1","mesh_batched_fanout.v1") and "require_complete" in inspect.signature(audit).parameters:
    r=audit(s,m,inputs,results,require_complete=False)
    r["completed_inspection_scope"]="only saved complete calls; full_run_passed is separate"
elif profile in ("mesh_batched_feed_forward.v1","mesh_batched_rms.v1","mesh_batched_fanout.v1"):
    # The first frozen batched auditor validates full runs. Do not reinterpret
    # a partial run or pass an unsupported partial-audit keyword to it.
    r=audit(s,m,inputs,results)
    r["completed_inspection_scope"]="full saved batch required by this frozen auditor"
else:
    r=audit(s,m,inputs,results,require_complete=False)
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
