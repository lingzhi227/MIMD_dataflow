"""Machine-readable stage inspection, actor diff and CPU body trace. No SDK import."""

import argparse, json, struct
from pathlib import Path
from ir import evaluate
from local_kernel import evaluate_kernel
from float32 import close


def main():
    p = argparse.ArgumentParser()
    p.add_argument("case")
    p.add_argument("--node")
    p.add_argument("--epoch", type=int, default=0)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--trace", action="store_true")
    p.add_argument(
        "--check-completed",
        action="store_true",
        help="Validate completed supported resident calls without claiming whole-run qualification",
    )
    p.add_argument("--limit", type=int, default=100)
    a = p.parse_args()
    root = Path(a.case)

    snapshot = {}
    raw_snapshot = {}

    def read(name):
        if name not in snapshot:
            raw_snapshot[name] = (
                (root / name).read_bytes() if (root / name).exists() else None
            )
            snapshot[name] = (
                json.loads(raw_snapshot[name])
                if raw_snapshot[name] is not None
                else None
            )
        return snapshot[name]

    s = read("schedule.json")
    output = {
        "build_stage": read("stage.json"),
        "sdk_stage": read("execution-stage.json"),
        "last_runtime_operation": read("runtime-stage.json"),
        "execution_success": (read("results.json") or {}).get("success"),
        "source": str(root / "source.cpp"),
        "clang_diagnostics": str(root / "00_clang_diagnostics.txt"),
        "native_command": read("native-command.json"),
        "native_stderr": (
            str(root / "native-stderr.txt")
            if (root / "native-stderr.txt").exists()
            else None
        ),
        "native_compile_log": (
            str(root / "native-compile.log")
            if (root / "native-compile.log").exists()
            else None
        ),
    }
    if a.check_completed and (s or {}).get("profile") not in (
        "mesh_normalized_fanout.v1",
        "mesh_attention.v1",
        "mesh_mlp.v1",
        "mesh_projection_residual_rms.v1",
        "mesh_feed_forward.v1",
        "mesh_prefill_tail.v1",
        "mesh_attention_tail.v1",
        "mesh_batched_rms.v1",
        "mesh_batched_fanout.v1",
        "mesh_batched_feed_forward.v1",
        "mesh_cache_attention.v1",
        "mesh_projected_cache.v1",
        "mesh_projected_cache_ffn.v1",
        "mesh_input_attention_mixed.v1",
    ):
        raise ValueError(
            "completed-call validation currently supports normalized fan-out, attention, MLP and projection/residual/RMS"
        )
    if s is None:
        print(json.dumps(output, indent=2))
        return
    if a.check_completed and read("results.json") is None:
        output["completed_call_diagnostic"] = dict(
            available=False,
            completed_calls=0,
            passed=False,
            reason="No completed SDK call has been saved yet",
        )
        output["diagnostic_is_full_qualification"] = False
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") in (
        "mesh_mlp.v1",
        "mesh_projection_residual_rms.v1",
        "mesh_feed_forward.v1",
        "mesh_prefill_tail.v1",
        "mesh_attention_tail.v1",
        "mesh_batched_rms.v1",
        "mesh_batched_fanout.v1",
        "mesh_batched_feed_forward.v1",
        "mesh_cache_attention.v1",
        "mesh_projected_cache.v1",
        "mesh_projected_cache_ffn.v1",
        "mesh_input_attention_mixed.v1",
    ):
        if s["profile"] == "mesh_projected_cache_ffn.v1":
            from projected_cache_ffn_debug import inspect
        elif s["profile"] == "mesh_projected_cache.v1":
            from projected_cache_debug import inspect
        elif s["profile"] == "mesh_cache_attention.v1":
            from cache_attention_debug import inspect
        elif s["profile"] == "mesh_batched_feed_forward.v1":
            from batched_ffn_debug import inspect
        elif s["profile"] == "mesh_batched_fanout.v1":
            from batched_fanout_debug import inspect
        elif s["profile"] == "mesh_batched_rms.v1":
            from batched_rms_debug import inspect
        elif s["profile"] == "mesh_mlp.v1":
            from mlp_debug import inspect
        elif s["profile"] == "mesh_input_attention_mixed.v1":
            from input_attention_mixed_debug import inspect
        elif s["profile"] == "mesh_attention_tail.v1":
            from attention_tail_debug import inspect
        elif s["profile"] == "mesh_prefill_tail.v1":
            from prefill_tail_debug import inspect
        elif s["profile"] == "mesh_feed_forward.v1":
            from feed_forward_debug import inspect
        else:
            from projection_residual_rms_debug import inspect

        if a.check_completed:
            import hashlib
            from frozen_inspection import completed

            output["completed_call_diagnostic"] = completed(root, read("results.json"))
            output["frozen_bundle_verified"] = True
            output["diagnostic_arithmetic_implementation"] = (
                "bundle/implementation (frozen)"
            )
            output["selected_view_implementation"] = (
                "current inspector, separately hashed below"
            )
            output["diagnostic_is_full_qualification"] = False
            output["diagnostic_results_sha256"] = hashlib.sha256(
                raw_snapshot["results.json"]
            ).hexdigest()
            output["diagnostic_implementation_sha256"] = {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in Path(__file__).parent.glob("*.py")
            }

        output.update(schedule=s)
        if a.node:
            output["selected"] = inspect(
                s, read("results.json"), a.node, a.epoch, a.step
            )
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_attention.v1":
        from attention_debug import inspect

        if a.check_completed:
            import hashlib
            import subprocess, sys
            from mesh_attention import plan
            from mesh_attention_sdk import audit_cases

            code = 'import sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from integrity import verify_bundle;verify_bundle(p)'
            subprocess.run(
                [sys.executable, "-c", code, str(root.resolve())],
                check=True,
                capture_output=True,
                text=True,
            )
            output["frozen_bundle_verified"] = True
            semantic = read("semantic.json")
            if s != plan(semantic):
                raise ValueError("completed-call schedule regeneration mismatch")
            output["completed_call_diagnostic"] = audit_cases(
                s,
                semantic,
                read("batches.json"),
                read("results.json"),
                require_complete=False,
            )
            output["diagnostic_is_full_qualification"] = False
            output["diagnostic_results_sha256"] = hashlib.sha256(
                raw_snapshot["results.json"]
            ).hexdigest()
            output["diagnostic_implementation_sha256"] = {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in Path(__file__).parent.glob("*.py")
            }

        output.update(schedule=s)
        if a.node:
            output["selected"] = inspect(
                s, read("results.json"), a.node, a.epoch, a.step
            )
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_score_softmax.v1":
        from score_softmax_debug import inspect

        output.update(schedule=s)
        if a.node:
            output["selected"] = inspect(
                s, read("results.json"), a.node, a.epoch, a.step
            )
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_device_matmul.v1":
        from device_matmul_debug import inspect

        output.update(schedule=s)
        if a.node:
            output["selected"] = inspect(
                s, read("results.json"), a.node, a.epoch, a.step
            )
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_score.v1":
        from score_debug import inspect

        output.update(schedule=s)
        if a.node:
            output["selected"] = inspect(
                s, read("results.json"), a.node, a.epoch, a.step
            )
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_pair_rotation.v1":
        from pair_rotation_debug import inspect

        output.update(schedule=s)
        if a.node:
            output["selected"] = inspect(
                s, read("results.json"), a.node, a.epoch, a.step
            )
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_swiglu.v1":
        from swiglu_debug import inspect

        output.update(schedule=s)
        if a.node:
            output["selected"] = inspect(
                s, read("results.json"), a.node, a.epoch, a.step
            )
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_normalized_fanout.v1":
        from normalized_fanout_debug import inspect

        if a.check_completed:
            import hashlib
            import subprocess, sys
            from mesh_normalized_fanout import plan
            from mesh_normalized_fanout_sdk import audit_cases

            code = 'import sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from integrity import verify_bundle;verify_bundle(p)'
            subprocess.run(
                [sys.executable, "-c", code, str(root.resolve())],
                check=True,
                capture_output=True,
                text=True,
            )
            output["frozen_bundle_verified"] = True
            semantic = read("semantic.json")
            if s != plan(semantic):
                raise ValueError("completed-call schedule regeneration mismatch")
            output["completed_call_diagnostic"] = audit_cases(
                s,
                semantic,
                read("batches.json"),
                read("results.json"),
                require_complete=False,
            )
            output["diagnostic_is_full_qualification"] = False
            output["diagnostic_results_sha256"] = hashlib.sha256(
                raw_snapshot["results.json"]
            ).hexdigest()
            output["diagnostic_implementation_sha256"] = {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in Path(__file__).parent.glob("*.py")
            }

        output.update(schedule=s)
        if a.node:
            output["selected"] = inspect(
                s, read("results.json"), a.node, a.epoch, a.step
            )
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_normalized_matmul.v1":
        from normalized_matmul_debug import inspect

        output.update(schedule=s)
        if a.node:
            output["selected"] = inspect(
                s, read("results.json"), a.node, a.epoch, a.step
            )
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_softmax.v1":
        from softmax_debug import inspect

        output.update(schedule=s)
        if a.node:
            output["selected"] = inspect(
                s, read("results.json"), a.node, a.epoch, a.step
            )
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_rms.v1":
        from rms_debug import inspect

        output.update(schedule=s)
        if a.node:
            output["selected"] = inspect(
                s, read("results.json"), a.node, a.epoch, a.step
            )
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_fft.v1":
        from fft_debug import inspect

        output.update(schedule=s)
        if a.node:
            output["selected"] = inspect(
                s, read("results.json"), a.node, a.epoch, a.step
            )
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_grouped_gemv.v1":
        from grouped_gemv_debug import inspect

        output.update(schedule=s)
        if a.node:
            output["selected"] = inspect(
                s, read("results.json"), a.node, a.epoch, a.step
            )
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_twohop.v1":
        from twohop_debug import inspect

        output.update(schedule=s)
        if a.node:
            output["selected"] = inspect(
                s, read("results.json"), a.node, a.epoch, a.step
            )
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_power.v1":
        import re

        output.update(schedule=s)
        if a.node:
            match = re.fullmatch(r"p(\d+)_(\d+)", a.node)
            if not match:
                raise ValueError("Power PE must be p<column>_<row>")
            col, row = map(int, match.groups())
            if not (
                0 <= col < s["cols"]
                and 0 <= row < s["rows"]
                and 0 <= a.epoch < s["epochs"]
            ):
                raise ValueError("Power PE/epoch bounds")
            selected = dict(node=a.node, epoch=a.epoch, observed=False)
            r = read("results.json")
            if r and a.epoch < len(r.get("diagnostics", [])):
                d = r["diagnostics"][a.epoch]
                selected.update(observed=True, **{k: v[row][col] for k, v in d.items()})
                attempts = d["power_attempts"][row][col][0]
                if not 0 <= attempts <= s["max_iterations"]:
                    raise ValueError("Power attempt count")
                selected["operator_witness_active"] = bool(attempts)
                selected["operator_witness_semantics"] = (
                    "Last pre-normalization operator application"
                    if attempts
                    else "Inactive: buffers may contain a preceding warm call"
                )
                selected["attempt_records"] = [
                    dict(
                        attempt=i + 1,
                        completed=i < d["cg_iterations"][row][col][0],
                        norm=d["cg_history"][row][col][i],
                        inverse=d["power_inverse"][row][col][i],
                    )
                    for i in range(attempts)
                ]
            output["selected"] = selected
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_cg.v1":
        import re

        output.update(schedule=s)
        if a.node:
            match = re.fullmatch(r"p(\d+)_(\d+)", a.node)
            if not match:
                raise ValueError("CG PE must be p<column>_<row>")
            col, row = map(int, match.groups())
            if not (
                0 <= col < s["cols"]
                and 0 <= row < s["rows"]
                and 0 <= a.epoch < s["epochs"]
            ):
                raise ValueError("CG PE/epoch bounds")
            selected = dict(node=a.node, epoch=a.epoch, observed=False)
            r = read("results.json")
            if r and a.epoch < len(r.get("diagnostics", [])):
                d = r["diagnostics"][a.epoch]
                selected.update(observed=True, **{k: v[row][col] for k, v in d.items()})
                selected["iteration_records"] = [
                    dict(
                        iteration=i + 1,
                        alpha=d["cg_scalars"][row][col][3 * i],
                        curvature=d["cg_scalars"][row][col][3 * i + 1],
                        beta=d["cg_scalars"][row][col][3 * i + 2],
                        residual_squared=d["cg_history"][row][col][i + 1],
                    )
                    for i in range(d["cg_iterations"][row][col][0])
                ]
                if s.get("solver") == "bicgstab":
                    for record in selected["iteration_records"]:
                        record["omega"] = record.pop("curvature")
                    attempts = d["bi_progress"][row][col][0]
                    if not 0 <= attempts <= s["max_iterations"]:
                        raise ValueError("BiCGStab attempt count")
                    selected["attempt_records"] = [
                        dict(
                            attempt=i + 1,
                            completed=i < d["cg_iterations"][row][col][0],
                            rho=d["bi_rhos"][row][col][i],
                            shadow_dot_v=d["bi_products"][row][col][3 * i],
                            s_squared=d["bi_ss"][row][col][i],
                            t_dot_s=d["bi_products"][row][col][3 * i + 1],
                            t_dot_t=d["bi_products"][row][col][3 * i + 2],
                        )
                        for i in range(attempts)
                    ]
                elif s.get("preconditioner") == "jacobi":
                    for i, record in enumerate(selected["iteration_records"]):
                        record["weighted_inner_product"] = d["cg_weights"][row][col][i]
            output["selected"] = selected
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_reduction.v1":
        import re
        import math
        from mesh_reduction_sdk import reference

        output.update(schedule=s)
        if a.node:
            match = re.fullmatch(r"p(\d+)_(\d+)", a.node)
            if not match:
                raise ValueError("Reduction PE must be p<column>_<row>")
            col, row = map(int, match.groups())
            if not (
                0 <= col < s["cols"]
                and 0 <= row < s["rows"]
                and 0 <= a.epoch < s["epochs"]
            ):
                raise ValueError("Reduction PE/epoch bounds")
            m = read("semantic.json")
            batch = read("batches.json")[a.epoch]
            arrays = [batch[n["host"]] for n in m["nodes"][:-2]]
            start = (row * s["cols"] + col) * s["local_length"]
            end = min(start + s["local_length"], s["N"])
            part = [v[start:end] for v in arrays]
            selected = dict(
                node=a.node,
                epoch=a.epoch,
                logical_interval=[min(start, s["N"]), end],
                valid_count=max(0, end - start),
                padding_count=s["local_length"] - max(0, end - start),
                observed=False,
            )
            selected["expected_global"] = reference(s["operation"], arrays)
            if s["operation"] == "dot":
                selected["expected_local_dot"] = reference("dot", part)
            else:
                peak = max(map(abs, arrays[0]))
                alpha = 1.0 if peak == 0 else 2.0 ** max(-126, math.frexp(peak)[1] - 1)
                selected.update(
                    expected_local_max=max(map(abs, part[0]), default=0.0),
                    expected_global_max=peak,
                    expected_scale=alpha,
                    expected_scaled_square_sum=math.fsum(
                        (v / alpha) ** 2 for v in part[0]
                    ),
                )
            device = read("results.json")
            if device and a.epoch < len(device.get("diagnostics", [])):
                d = device["diagnostics"][a.epoch]
                selected.update(
                    observed=True,
                    replica=d["replicas"][row][col],
                    witness=d["witness"][row][col],
                    progress=d["progress"][row][col],
                    timing=d["timing"][row][col],
                )
            output["selected"] = selected
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_spmv.v1":
        import re
        import math

        output.update(schedule=s)
        if a.node:
            match = re.fullmatch(r"p(\d+)_(\d+)", a.node)
            if not match:
                raise ValueError("SpMV PE must be p<column>_<row>")
            col, row = map(int, match.groups())
            if not (
                0 <= col < s["cols"]
                and 0 <= row < s["rows"]
                and 0 <= a.epoch < s["epochs"]
            ):
                raise ValueError("SpMV PE/epoch bounds")
            pack = read("sparse-packing.json")[a.epoch]["tiles"][row][col]
            selected = {
                "node": a.node,
                "epoch": a.epoch,
                "local_counts": {
                    name: pack[name][0]
                    for name in ("local_nnz", "local_nnz_cols", "local_nnz_rows")
                },
                "capacity": s["capacity"],
                "observed": False,
            }
            batch = read("batches.json")[a.epoch]
            m = read("semantic.json")
            values, indices, offsets, x = [batch[n["host"]] for n in m["nodes"][:4]]
            terms = {}
            g = s["geometry"]
            for c in range(s["N"]):
                if c // g["block_cols"] != col:
                    continue
                for p in range(offsets[c], offsets[c + 1]):
                    if indices[p] // g["block_rows"] == row:
                        terms.setdefault(indices[p], []).append(
                            float(values[p]) * float(x[c])
                        )
            occupied = sorted(terms)
            selected["sample_rows"] = (
                [] if not occupied else [occupied[0], occupied[-1]]
            )
            selected["expected_partial"] = (
                []
                if not occupied
                else [math.fsum(terms[r]) for r in selected["sample_rows"]]
            )
            device = read("results.json")
            if device and a.epoch < len(device.get("diagnostics", [])):
                d = device["diagnostics"][a.epoch]
                selected.update(
                    observed=True,
                    progress=d["progress"][row][col],
                    partial=d["partial"][row][col],
                    timing=d["timing"][row][col],
                )
            output["selected"] = selected
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_qr.v1":
        import re
        from qr_schedule import rotations
        from mesh_qr_sdk import audit_rotations

        output.update(schedule=s)
        if a.node:
            match = re.fullmatch(r"p(\d+)_(\d+)", a.node)
            if not match:
                raise ValueError("QR PE must be p<column>_<row>")
            col, row = map(int, match.groups())
            if not (
                0 <= col < s["cols"]
                and 0 <= row < s["rows"]
                and 0 <= a.epoch < s["epochs"]
            ):
                raise ValueError("QR PE/epoch bounds")
            kinds = rotations(s["rows"], s["cols"], s["Nt"], col, row)
            if not 0 <= a.step < len(kinds):
                raise ValueError("QR rotation bounds")
            selected = {
                "node": a.node,
                "epoch": a.epoch,
                "rotation": a.step,
                "expected_role": kinds[a.step],
                "expected_total": len(kinds),
                "role_names": {
                    "1": "local row pair",
                    "2": "upper neighbor row",
                    "3": "lower neighbor row",
                },
            }
            if a.trace:
                selected["symbolic_rotation_roles"] = kinds[: a.limit]
            device = read("results.json")
            if device and a.epoch < len(device["diagnostics"]):
                diag = device["diagnostics"][a.epoch]
                samples = diag["witnesses"][row][col]
                total, done = diag["progress"][row][col]
                selected.update(
                    observed_total=total,
                    completed=done,
                    timestamp_words=diag["timing"][row][col],
                )
                if s.get("instrumentation", "sampled") == "counters":
                    selected["sample_status"] = "disabled in counters mode"
                    selected["symbolic_count_passed"] = total == len(kinds)
                    output["selected"] = selected
                    print(json.dumps(output, indent=2))
                    return
                try:
                    selected["sample_audit"] = audit_rotations(samples, total, kinds)
                except (ValueError, AssertionError) as error:
                    selected["sample_audit_error"] = str(error)
                matches = [w for w in samples if w[0] != 0 and w[1] == a.step]
                selected["sample_status"] = (
                    "retained" if matches else "not retained by bounded sampler"
                )
                if matches:
                    kind, serial, c, sine, u, v, top, bottom = matches[0]
                    selected["witness"] = dict(
                        kind=kind,
                        serial=serial,
                        cos=c,
                        sin=sine,
                        a=u,
                        b=v,
                        observed_top=top,
                        observed_bottom=bottom,
                        expected_top=c * u - sine * v if kind in (1, 2) else None,
                        expected_bottom=sine * u + c * v if kind in (1, 3) else None,
                    )
            output["selected"] = selected
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") in ("mesh_cholesky.v1", "mesh_lu.v1"):
        import numpy as np
        import re

        if s["profile"] == "mesh_lu.v1":
            from mesh_lu_sdk import checkpoint_reference
        else:
            from mesh_cholesky_sdk import checkpoint_reference

        output.update(schedule=s)
        if a.node:
            match = re.fullmatch(r"p(\d+)_(\d+)", a.node)
            if not match:
                raise ValueError("Factorization PE must be p<column>_<row>")
            col, row = map(int, match.groups())
            if not (
                0 <= col < s["P"]
                and 0 <= row < s["P"]
                and 0 <= a.epoch < s["epochs"]
                and 0 <= a.step < s["N"]
            ):
                raise ValueError("Factorization PE/pivot/epoch bounds")
            m = read("semantic.json")
            batch = read("batches.json")[a.epoch]
            matrix = np.asarray(batch[m["nodes"][0]["host"]]).reshape(s["N"], s["N"])
            expected = checkpoint_reference(matrix, s["P"])[row, col, a.step]
            selected = {
                "node": a.node,
                "epoch": a.epoch,
                "pivot": a.step,
                "active_at_pivot": (
                    a.step < (min(col, row) + 1) * s["Nt"]
                    if s["profile"] == "mesh_lu.v1"
                    else col <= row and a.step < (col + 1) * s["Nt"]
                ),
                "expected_tile_corners": expected.tolist(),
            }
            device = read("results.json")
            if device and a.epoch < len(device["diagnostics"]):
                diag = device["diagnostics"][a.epoch]
                selected.update(
                    progress=diag["progress"][row][col],
                    timestamp_words=diag["timing"][row][col],
                )
                if "checkpoints" in diag:
                    actual = diag["checkpoints"][row][col][a.step]
                    selected.update(
                        actual_tile_corners=actual,
                        fixed_accuracy_passed=bool(
                            np.allclose(actual, expected, rtol=3e-5, atol=3e-6)
                        ),
                    )
            output["selected"] = selected
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") in ("mesh_gemm.v1", "mesh_cannon.v1"):
        import numpy as np
        import re
        from roundoff import check_matrix_roundoff

        cannon = s["profile"] == "mesh_cannon.v1"
        order = "C" if cannon else "F"
        output.update(schedule=s)
        if a.node:
            match = re.fullmatch(r"p(\d+)_(\d+)", a.node)
            if not match:
                raise ValueError("SUMMA PE must be p<column>_<row>")
            col, row = map(int, match.groups())
            if not (
                0 <= col < s["P"]
                and 0 <= row < s["P"]
                and 0 <= a.step < s["P"]
                and 0 <= a.epoch < s["epochs"]
            ):
                raise ValueError("SUMMA PE/round/epoch bounds")
            m = read("semantic.json")
            batch = read("batches.json")[a.epoch]
            left, right = [
                np.asarray(batch[n["host"]]).reshape(n["shape"]) for n in m["nodes"][:2]
            ]
            mt, nt = s["Mt"], s["Nt"]
            end = (a.step + 1) * s["Kt"]
            indices = (
                [
                    q
                    for rstep in range(a.step + 1)
                    for q in range(
                        ((row + col + rstep) % s["P"]) * s["Kt"],
                        ((row + col + rstep) % s["P"] + 1) * s["Kt"],
                    )
                ]
                if cannon
                else list(range(end))
            )
            panel_a = left[row * mt : (row + 1) * mt, indices]
            panel_b = right[indices, col * nt : (col + 1) * nt]
            expected = panel_a @ panel_b
            output["selected"] = {
                "node": a.node,
                "epoch": a.epoch,
                "round": a.step,
                "shape": [mt, nt],
                "accumulated_K": end,
                "expected_tile": expected.ravel(order=order)[: a.limit].tolist(),
                "tile_memory_order": order,
                "global_K_indices": indices,
            }
            if not cannon:
                output["selected"]["expected_column_major"] = output["selected"][
                    "expected_tile"
                ]
            device = read("results.json")
            if device and a.epoch < len(device["diagnostics"]):
                diag = device["diagnostics"][a.epoch]
                actual = np.asarray(diag["history"][row][col][a.step]).reshape(
                    mt, nt, order=order
                )
                output["selected"].update(
                    actual_tile=actual.ravel(order=order)[: a.limit].tolist(),
                    timestamp_words=diag["timing"][row][col][a.step],
                )
                if not cannon:
                    output["selected"]["actual_column_major"] = output["selected"][
                        "actual_tile"
                    ]
                if cannon:
                    output["selected"].update(
                        block_witness=diag["witness"][row][col][a.step],
                        progress=diag["progress"][row][col],
                        queue_last=diag["queue_last"][row][col],
                        total_timestamp_words=diag["total_timing"][row][col],
                    )
                try:
                    output["selected"]["roundoff"] = check_matrix_roundoff(
                        panel_a, panel_b, actual, expected
                    )
                    output["selected"]["roundoff_passed"] = True
                except ValueError as error:
                    output["selected"].update(
                        roundoff_passed=False, diagnostic=str(error)
                    )
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "mesh_gemv.v1":
        output.update(schedule=s)
        if a.node:
            import re
            from float32 import f32

            match = re.fullmatch(r"p(\d+)_(\d+)", a.node)
            if not match:
                raise ValueError("mesh PE identifier must be p<column>_<row>")
            col, row = map(int, match.groups())
            if not (
                0 <= col < s["kernel_cols"]
                and 0 <= row < s["kernel_rows"]
                and 0 <= a.epoch < s["epochs"]
            ):
                raise ValueError("mesh PE/epoch out of range")
            m = read("semantic.json")
            batch = read("batches.json")[a.epoch]
            matrix = batch[m["nodes"][0]["host"]]
            vector = batch[m["nodes"][1]["host"]]
            mt, nt = s["Mt"], s["Nt"]
            expected = []
            for i in range(mt):
                value = 0.0
                for j in range(nt):
                    index = (row * mt + i) * s["matrix_cols"] + col * nt + j
                    value = f32(value + f32(matrix[index] * vector[col * nt + j]))
                expected.append(value)
            output["selected"] = {
                "node": a.node,
                "epoch": a.epoch,
                "expected_partial": expected,
            }
            device = read("results.json")
            if device and a.epoch < len(device["diagnostics"]):
                d = device["diagnostics"][a.epoch]
                actual = d["partial"][row][col]
                output["selected"].update(
                    actual_partial=actual,
                    matches=close(actual, expected),
                    vector_partition=d["x_tile"][row][col],
                    compute_timestamp_words=d["compute_time"][row][col],
                )
        print(json.dumps(output, indent=2))
        return
    if s.get("profile") == "grid.v1":
        output.update(
            grid=s["grid"], actors=s["nodes"], numeric_lowering=s["numeric_lowering"]
        )
        if a.node:
            from grid_ir import simulate

            by = {n["id"]: n for n in s["nodes"]}
            if a.node not in by:
                raise ValueError("unknown PE")
            if not 0 <= a.epoch < s["epochs"] or not 0 <= a.step < s["grid"]["steps"]:
                raise ValueError("epoch/step out of range")
            z = s["grid"]["z"]
            _, history = simulate(
                read("semantic.json"), read("batches.json")[a.epoch], True
            )
            expected = history[a.node][a.step * z : (a.step + 1) * z]
            output["selected"] = {
                "pe": by[a.node],
                "epoch": a.epoch,
                "step": a.step,
                "expected": expected,
            }
            device = read("results.json")
            if device:
                offset = (a.epoch * s["grid"]["steps"] + a.step) * z
                raw = device["diagnostics"][a.node]["history"][offset : offset + z]
                actual = [struct.unpack("f", struct.pack("I", v))[0] for v in raw]
                output["selected"].update(
                    actual=actual, matches=close(actual, expected)
                )
            if a.trace:
                output["selected"]["step_history"] = history[a.node][: a.limit * z]
        print(json.dumps(output, indent=2))
        return
    output["actors"] = [
        {
            "id": n["id"],
            "op": n["op"],
            "shape": n["shape"],
            "place": n["place"],
            "source_line": n["line"],
            "inputs": n["inputs"],
            "memory": n["memory"],
        }
        for n in s["nodes"]
    ]
    if a.node:
        m = read("semantic.json")
        by = {n["id"]: n for n in s["nodes"]}
        if a.node not in by:
            raise ValueError("unknown actor " + a.node)
        if not 0 <= a.epoch < s["epochs"]:
            raise ValueError("epoch out of range")
        n = by[a.node]
        _, history = evaluate(dict(m, nodes=s["nodes"]), read("batches.json"))
        size = n["output_size"]
        expected = history[a.node][a.epoch * size : (a.epoch + 1) * size]
        output["selected"] = {"node": n, "expected": expected}
        device = read("results.json")
        if device and a.node in device.get("diagnostics", {}):
            raw = device["diagnostics"][a.node]["history"]
            actual = [
                struct.unpack("f", struct.pack("I", v))[0]
                for v in raw[a.epoch * size : (a.epoch + 1) * size]
            ]
            output["selected"].update(actual=actual, matches=close(actual, expected))
        if a.trace:
            if n["op"] != "kernel":
                raise ValueError("body trace requires kernel actor")
            inputs = [
                history[key][
                    a.epoch
                    * by[key]["output_size"] : (a.epoch + 1)
                    * by[key]["output_size"]
                ]
                for key in n["inputs"]
            ]
            events = []
            evaluate_kernel(n["body"], inputs, events)
            output["selected"].update(
                symbols=n["body"]["symbols"],
                trace=events[: a.limit],
                trace_events=len(events),
                trace_truncated=len(events) > a.limit,
            )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
