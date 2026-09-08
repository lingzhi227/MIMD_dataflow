"""Audit generated provenance, device values, resident state and lifecycle."""

import json, sys, struct
from pathlib import Path
from frontend import check
from ir import evaluate
from backend import actor
from planner import plan
from integrity import verify_bundle
from float32 import close


def audit(root):
    root = Path(root)
    manifest = verify_bundle(root)
    m = json.loads((root / "semantic.json").read_text())
    s = json.loads((root / "schedule.json").read_text())
    if s.get("profile") == "mesh_input_attention_mixed.v1":
        from mesh_input_attention_mixed_sdk import audit as mixed_audit

        return mixed_audit(root)
    if s.get("profile") == "mesh_attention_tail.v1":
        from mesh_attention_tail_sdk import audit as attention_tail_audit

        return attention_tail_audit(root)
    if s.get("profile") == "mesh_prefill_tail.v1":
        from mesh_prefill_tail_sdk import audit as tail_audit

        return tail_audit(root)
    if s.get("profile") == "mesh_feed_forward.v1":
        from mesh_feed_forward_sdk import audit as feed_forward_audit

        return feed_forward_audit(root)
    if s.get("profile") == "mesh_projection_residual_rms.v1":
        from mesh_projection_residual_rms_sdk import audit as composition_audit

        return composition_audit(root)
    if s.get("profile") == "mesh_mlp.v1":
        from mesh_mlp_sdk import audit as mlp_audit

        return mlp_audit(root)
    if s.get("profile") == "mesh_attention.v1":
        from mesh_attention_sdk import audit as attention_audit

        return attention_audit(root)
    if s.get("profile") == "mesh_score_softmax.v1":
        from mesh_score_softmax_sdk import audit as resident_audit

        return resident_audit(root)
    if s.get("profile") == "mesh_device_matmul.v1":
        from mesh_device_matmul_sdk import audit as device_audit

        return device_audit(root)
    if s.get("profile") == "mesh_score.v1":
        from mesh_score_sdk import audit as score_audit

        return score_audit(root)
    if s.get("profile") == "mesh_pair_rotation.v1":
        from mesh_pair_rotation_sdk import audit as pair_audit

        return pair_audit(root)
    if s.get("profile") == "mesh_swiglu.v1":
        from mesh_swiglu_sdk import audit as gating_audit

        return gating_audit(root)
    if s.get("profile") == "mesh_normalized_fanout.v1":
        from mesh_normalized_fanout_sdk import audit as fanout_audit

        return fanout_audit(root)
    if s.get("profile") == "mesh_normalized_matmul.v1":
        from mesh_normalized_matmul_sdk import audit as resident_audit

        return resident_audit(root)
    if s.get("profile") == "mesh_softmax.v1":
        from mesh_softmax_sdk import audit as softmax_audit

        return softmax_audit(root)
    if s.get("profile") == "mesh_rms.v1":
        from mesh_rms_sdk import audit as rms_audit

        return rms_audit(root)
    if s.get("profile") == "mesh_fft.v1":
        from mesh_fft_sdk import audit as fft_audit

        return fft_audit(root)
    if s.get("profile") == "mesh_grouped_gemv.v1":
        from grouped_gemv_sdk import audit as grouped_audit

        return grouped_audit(root)
    if s.get("profile") == "mesh_twohop.v1":
        from twohop_sdk import audit as half_audit

        return half_audit(root)
    if s.get("profile") == "mesh_reduction.v1":
        from mesh_reduction_sdk import audit as reduction_audit

        return reduction_audit(root)
    if s.get("profile") == "mesh_power.v1":
        from power_audit import audit as power_audit

        return power_audit(root)
    if s.get("profile") == "mesh_cg.v1":
        from mesh_cg_sdk import audit as cg_audit

        return cg_audit(root, manifest)
    if s.get("profile") == "mesh_spmv.v1":
        from mesh_spmv_sdk import audit as sparse_audit

        return sparse_audit(root, manifest)
    if s.get("profile") == "mesh_qr.v1":
        from mesh_qr_sdk import audit as qr_audit

        return qr_audit(root, manifest)
    if s.get("profile") == "mesh_lu.v1":
        from mesh_lu_sdk import audit as lu_audit

        return lu_audit(root, manifest)
    if s.get("profile") == "mesh_cholesky.v1":
        from mesh_cholesky_sdk import audit as chol_audit

        return chol_audit(root, manifest)
    if s.get("profile") in ("mesh_gemm.v1", "mesh_cannon.v1"):
        from mesh_gemm_sdk import audit as mesh_audit

        return mesh_audit(root, manifest)
    if s.get("profile") == "mesh_gemv.v1":
        from mesh_gemv_sdk import audit as mesh_audit

        return mesh_audit(root, manifest)
    if s.get("profile") == "grid.v1":
        from grid_validate import audit as grid_audit

        return grid_audit(root, manifest)
    b = json.loads((root / "batches.json").read_text())
    r = json.loads((root / "results.json").read_text())
    check(plan(m, s["matmul_partitions"]) == s, "schedule regeneration mismatch")
    expected, _ = evaluate(m, b)
    scheduled, history = evaluate(dict(m, nodes=s["nodes"]), b)
    check(close(scheduled, expected), "lowering semantic mismatch")
    check(r["success"], "execution incomplete")
    check(close(r["cases"], expected), "device output mismatch")
    by = {n["id"]: n for n in s["nodes"]}
    observations = 0
    for n in s["nodes"]:
        name = n["id"]
        check(
            (root / (name + ".csl")).read_text() == actor(n, by, s["epochs"]),
            "CSL regeneration mismatch " + name,
        )
        if n["op"] == "fork":
            history[name] = history[n["inputs"][0]]
        d = r["diagnostics"][name]
        actual = [struct.unpack("f", struct.pack("I", v))[0] for v in d["history"]]
        check(close(actual, history[name]), "intermediate mismatch " + name)
        observations += len(actual)
        required = {
            "received_a": n["wire_input_sizes"][0] * s["epochs"],
            "produced": n["output_size"] * s["epochs"],
            "completed": n["wire_output_size"] * s["epochs"] * n["send_ports"],
            "epochs": s["epochs"],
            "inflight": 0,
        }
        if len(n["input_sizes"]) == 2:
            required["received_b"] = n["wire_input_sizes"][1] * s["epochs"]
        for field, value in required.items():
            check(d[field] == [value], "lifecycle " + name + "." + field)
            observations += 1
        if n["op"] == "accumulate":
            actual = [struct.unpack("f", struct.pack("I", v))[0] for v in d["state"]]
            check(close(actual, history[name][-n["output_size"] :]), "resident state")
            observations += len(actual)
    result = {
        "passed": True,
        "output_values": manifest["expected_output_values"],
        "internal_observations": observations,
        "epochs": s["epochs"],
        "actors": len(s["nodes"]),
        "source_sha256": manifest["source_sha256"],
    }
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(sys.argv[1]), indent=2))
