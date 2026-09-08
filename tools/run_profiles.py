"""Per-kernel fresh builds, staged artifacts and independent application checks."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, subprocess, sys, os, hashlib, platform, math
import numpy as np
from pathlib import Path

ROOT = repository_root(__file__)
sys.path.insert(0, str(ROOT / "lib"))
from compile import build
from validate import audit
from sdk_process import run_sdk
from fixtures import batches, check_application
from native_transport import parse_outputs


def numerical_summary(case):
    native = case.get("native_application_checks", [])
    device = case.get("device_application_checks", [])
    normwise_limits = {
        "composed-attention-ffn-half-normwise-v1": (0.02, 0.03),
        "shared-gamma-input-attention-tail-half-normwise-v1": (0.02, 0.03),
        "fft-f32-normwise-v1": (2e-5, 3e-5),
        "rms-half-normwise-v1": (0.01, 0.015),
        "softmax-half-normwise-v1": (0.01, 0.015),
        "normalized-matmul-half-normwise-v1": (0.015, 0.02),
        "normalized-fanout-half-normwise-v1": (0.015, 0.02),
        "score-half-normwise-v1": (0.015, 0.02),
        "device-matmul-half-normwise-v1": (0.015, 0.02),
        "score-softmax-half-normwise-v1": (0.015, 0.02),
        "attention-half-normwise-v1": (0.02, 0.025),
        "rectangular-mlp-half-normwise-v1": (0.02, 0.03),
        "projection-residual-rms-half-normwise-v1": (0.02, 0.03),
        "normalized-feed-forward-half-normwise-v1": (0.02, 0.03),
        "batched-feed-forward-half-normwise-v1": (0.02, 0.03),
        "supplied-cache-attention-half-normwise-v1": (0.02, 0.03),
        "projected-cache-half-normwise-v1": (0.02, 0.03),
        "supplied-attention-tail-half-normwise-v1": (0.02, 0.03),
        "supplied-qkv-attention-tail-half-normwise-v1": (0.02, 0.03),
    }
    contract = (
        native[0].get("contract") if native and isinstance(native[0], dict) else None
    )
    if contract == "pair-rotation-half-v1":
        if any(
            not isinstance(v, dict) or v.get("contract") != contract
            for v in native + device
        ):
            raise ValueError("mixed pair rotation contracts")
        native_ok = all(v["fixed_accuracy_passed"] for v in native)
        device_ok = all(v["fixed_accuracy_passed"] for v in device) if device else None
        return dict(
            contract=contract,
            native_fixed_accuracy_passed=native_ok,
            device_fixed_accuracy_passed=device_ok,
            device_audit_passed=case.get("audit", {}).get("passed"),
            fixed_accuracy_passed=native_ok
            and device_ok is not False
            and case.get("audit", {}).get("passed") is not False,
            componentwise_product_magnitude_term=0.0015,
            absolute_rounding_term=2**-23,
            scope="Per output: .0015*(abs(product0)+abs(product1))+2^-23; cancellation-aware, not output-relative or f32 tolerance",
        )
    if contract == "gated-activation-half-v1":
        if any(
            not isinstance(v, dict) or v.get("contract") != contract
            for v in native + device
        ):
            raise ValueError("mixed gated activation contracts")
        native_ok = all(v["fixed_accuracy_passed"] for v in native)
        device_ok = all(v["fixed_accuracy_passed"] for v in device) if device else None
        return dict(
            contract=contract,
            native_fixed_accuracy_passed=native_ok,
            device_fixed_accuracy_passed=device_ok,
            device_audit_passed=case.get("audit", {}).get("passed"),
            fixed_accuracy_passed=native_ok
            and device_ok is not False
            and case.get("audit", {}).get("passed") is not False,
            componentwise_relative_term=0.004,
            absolute_rounding_term="2^-24*(1+abs(up))",
            scope="bounded half gating, including stored activation and final product rounding; no f32 tolerance claim",
        )
    if contract in normwise_limits and all(
        isinstance(v, dict) and v.get("contract") == contract for v in native
    ):
        if any(
            not isinstance(v, dict) or v.get("contract") != contract for v in device
        ):
            raise ValueError("mixed normwise numerical contracts")
        if contract == "shared-gamma-input-attention-tail-half-normwise-v1":
            branches = (
                "input_normalized",
                "q_raw",
                "k_raw",
                "v_raw",
                "q",
                "k",
                "v",
                "score",
                "probability",
                "attention",
                "projection",
                "mlp_delta",
            )
            if any(
                not v.get(k, {}).get("fixed_accuracy_passed")
                for v in native + device
                for k in branches
            ):
                raise ValueError(
                    "Input attention requires all original-input branch gates"
                )
            if any(
                not v[k].get("local_pair_rounding_passed")
                for v in native + device
                for k in ("q", "k")
            ):
                raise ValueError("Input attention requires local pair rounding checks")
        if contract == "normalized-feed-forward-half-normwise-v1":
            if any(
                not isinstance(v.get("mlp_delta"), dict)
                or not v["mlp_delta"].get("fixed_accuracy_passed")
                for v in native + device
            ):
                raise ValueError("FFN requires a passing separately observed MLP delta")
        if contract == "supplied-qkv-attention-tail-half-normwise-v1":
            if any(
                not v.get(k, {}).get("fixed_accuracy_passed")
                for v in native + device
                for k in (
                    "score",
                    "probability",
                    "attention",
                    "projection",
                    "mlp_delta",
                )
            ):
                raise ValueError(
                    "Attention tail requires five separately observed passing branches"
                )
        if contract == "supplied-attention-tail-half-normwise-v1":
            if any(
                not v.get(k, {}).get("fixed_accuracy_passed")
                for v in native + device
                for k in ("projection", "mlp_delta")
            ):
                raise ValueError(
                    "Tail requires separately observed passing projection and MLP delta"
                )
        if contract == "supplied-cache-attention-half-normwise-v1":
            if any(
                not v.get("all_stage_gates")
                or set(v.get("stages", {}))
                != {"score", "probability", "context", "delta", "result"}
                or v.get("max_row_mass_error") is None
                or v["max_row_mass_error"] > 0.01
                for v in native + device
            ):
                raise ValueError(
                    "Cache attention requires all original-input stages and probability mass"
                )
        if contract == "projected-cache-half-normwise-v1":
            stages = {
                "normalized",
                "query",
                "key_projection",
                "value_projection",
                "rotated_query",
                "rotated_key",
                "score",
                "probability",
                "context",
                "delta",
                "result",
            }
            for v in native + device:
                if (
                    v.get("all_stage_gates") is not True
                    or set(v.get("metrics", {})) != stages
                    or v.get("limits")
                    != dict(relative_l2=0.02, relative_peak=0.03, row_mass=0.01)
                ):
                    raise ValueError(
                        "Projected cache requires eleven observed stages and fixed limits"
                    )
                mass = v.get("max_probability_mass_error")
                if (
                    type(mass) not in (int, float)
                    or not math.isfinite(mass)
                    or not 0 <= mass <= 0.01
                ):
                    raise ValueError("Projected cache probability mass")
                for metric in v["metrics"].values():
                    for key, limit in (("relative_l2", 0.02), ("relative_peak", 0.03)):
                        value = metric.get(key)
                        if (
                            type(value) not in (int, float)
                            or not math.isfinite(value)
                            or not 0 <= value <= limit
                        ):
                            raise ValueError("Projected cache stage numerical gate")
        native_fixed = all(v["fixed_accuracy_passed"] for v in native)
        device_fixed = (
            all(v["fixed_accuracy_passed"] for v in device) if device else None
        )
        return dict(
            contract=contract,
            native_fixed_accuracy_passed=native_fixed,
            device_fixed_accuracy_passed=device_fixed,
            device_audit_passed=case.get("audit", {}).get("passed"),
            fixed_accuracy_passed=native_fixed
            and device_fixed is not False
            and case.get("audit", {}).get("passed") is not False,
            relative_l2_limit=normwise_limits[contract][0],
            max_error_over_reference_peak_limit=normwise_limits[contract][1],
            zero_reference_requires_exact_zero=contract
            not in ("softmax-half-normwise-v1", "score-softmax-half-normwise-v1"),
            **(
                {
                    "max_row_mass_error_limit": 0.01,
                    "nonnegative_probability_required": True,
                }
                if contract
                in ("softmax-half-normwise-v1", "score-softmax-half-normwise-v1")
                else {}
            ),
            per_component_accuracy_not_implied=True,
        )
    if native and all(
        isinstance(v, dict) and v.get("contract") == "binary16-fma-relaxed-v1"
        for v in native
    ):
        if any(
            not isinstance(v, dict) or v.get("contract") != "binary16-fma-relaxed-v1"
            for v in device
        ):
            raise ValueError("mixed half numerical contracts")
        return dict(
            contract="binary16-fma-relaxed-v1",
            native_forward_bound_passed=all(v["forward_bound_passed"] for v in native),
            device_forward_bound_passed=(
                all(v["forward_bound_passed"] for v in device) if device else None
            ),
            device_scheduled_bits_passed=case.get("audit", {}).get("passed"),
            native_f32_accuracy_not_implied=True,
        )
    if native and all(
        isinstance(v, dict)
        and "contract" in v
        and "fixed_accuracy_passed" in v
        and "roundoff_passed" not in v
        for v in native
    ):
        contracts = {v["contract"] for v in native}
        if len(contracts) != 1 or any(
            not isinstance(v, dict)
            or v.get("contract") not in contracts
            or "fixed_accuracy_passed" not in v
            for v in device
        ):
            raise ValueError("mixed numerical contracts")
        if contract == "projected-cache-half-normwise-v1":
            stages = {
                "normalized",
                "query",
                "key_projection",
                "value_projection",
                "rotated_query",
                "rotated_key",
                "score",
                "probability",
                "context",
                "delta",
                "result",
            }
            for v in native + device:
                if (
                    v.get("all_stage_gates") is not True
                    or set(v.get("metrics", {})) != stages
                    or v.get("limits")
                    != dict(relative_l2=0.02, relative_peak=0.03, row_mass=0.01)
                ):
                    raise ValueError(
                        "Projected cache requires eleven observed stages and fixed limits"
                    )
                mass = v.get("max_probability_mass_error")
                if (
                    type(mass) not in (int, float)
                    or not math.isfinite(mass)
                    or not 0 <= mass <= 0.01
                ):
                    raise ValueError("Projected cache probability mass")
                for metric in v["metrics"].values():
                    for key, limit in (("relative_l2", 0.02), ("relative_peak", 0.03)):
                        value = metric.get(key)
                        if (
                            type(value) not in (int, float)
                            or not math.isfinite(value)
                            or not 0 <= value <= limit
                        ):
                            raise ValueError("Projected cache stage numerical gate")
        native_fixed = all(v["fixed_accuracy_passed"] for v in native)
        device_fixed = (
            all(v["fixed_accuracy_passed"] for v in device) if device else None
        )
        audit_fixed = case.get("audit", {}).get("fixed_accuracy_passed")
        return {
            "contract": next(iter(contracts)),
            "native_fixed_accuracy_passed": native_fixed,
            "device_fixed_accuracy_passed": device_fixed,
            "device_audit_fixed_accuracy_passed": audit_fixed,
            "fixed_accuracy_passed": native_fixed
            and device_fixed is not False
            and audit_fixed is not False,
            "rtol": 3e-5,
            "atol": 3e-6,
        }
    if not native or not all(
        isinstance(v, dict) and "roundoff_passed" in v for v in native
    ):
        return None
    native_fixed = all(v["fixed_accuracy_passed"] for v in native)
    device_fixed = all(v["fixed_accuracy_passed"] for v in device) if device else None
    audit_fixed = case.get("audit", {}).get("fixed_accuracy_passed")
    return {
        "contract": "componentwise-f32-dot-v1",
        "arithmetic_roundoff_passed": all(v["roundoff_passed"] for v in native + device)
        and case.get("audit", {}).get("arithmetic_roundoff_passed") is not False,
        "native_fixed_accuracy_passed": native_fixed,
        "device_fixed_accuracy_passed": device_fixed,
        "device_per_round_and_final_fixed_accuracy_passed": audit_fixed,
        "fixed_accuracy_passed": native_fixed
        and device_fixed is not False
        and audit_fixed is not False,
        "rtol": 3e-5,
        "atol": 3e-6,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sdk", action="store_true")
    p.add_argument(
        "--sdk-suppress-trace",
        action="store_true",
        help="Mesh profiles: suppress raw instruction traces; retain core and exported diagnostics",
    )
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--select", help="Substring of project/kernel")
    selection.add_argument("--select-exact", help="One exact project/kernel key")
    p.add_argument(
        "--sdk-threads",
        type=int,
        help="Simulator worker count; recorded with runtime options",
    )
    p.add_argument(
        "--instrumentation",
        choices=("sampled", "counters"),
        help="Supported profiles, including FFT: sampled internal states or lean counters; final numerical validation in both modes",
    )
    p.add_argument("--sdk-timeout", type=int, default=240)
    a = p.parse_args()
    if a.sdk and not all(os.environ.get(k) for k in ("PRAGMA_SDK_IMAGE", "PRAGMA_CS_PYTHON")):
        p.error("--sdk requires PRAGMA_SDK_IMAGE and PRAGMA_CS_PYTHON; see ../docs/REPRODUCING.md")
    if a.sdk_timeout <= 0 or (a.sdk_threads is not None and a.sdk_threads <= 0):
        p.error("SDK timeout and worker count must be positive")
    selected = [
        item
        for item in json.loads((ROOT / "benchmarks/catalog.json").read_text())
        if (not a.select or a.select in item["project"] + "/" + item["kernel"])
        and (
            not a.select_exact
            or a.select_exact == item["project"] + "/" + item["kernel"]
        )
    ]
    if not selected:
        p.error("No catalog profile matches the selection")
    run = "run-" + datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    report = {
        "run": run,
        "sdk": a.sdk,
        "instrumentation": a.instrumentation,
        "selection": {"substring": a.select, "exact": a.select_exact},
        "sdk_runtime_options": (
            {
                "suppress_trace": a.sdk_suppress_trace,
                "num_threads": a.sdk_threads or 16,
                "dump_core": True,
            }
            if a.sdk_suppress_trace or a.sdk_threads is not None
            else None
        ),
        "sdk_timeout_seconds": a.sdk_timeout,
        "success": False,
        "cases": [],
    }
    report["oracle"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rtol": 3e-5,
        "atol": 3e-6,
        "input_seed_base": 911,
        "additional_contract_by_fixture_prefix": {
            "composed_ffn:": "resident attention/FFN; eighteen original-input native/device stage gates, 2% L2 / 3% peak, probability mass 1%; source-local-compute performance scope",
            "mesh_gemm:": "componentwise-f32-dot-v1; fixed accuracy reported separately",
            "distributed_fft:": "fft-f32-normwise-v1; not per-component accuracy",
            "pair_rotation:": "pair-rotation-half-v1; cancellation-aware product-magnitude criterion",
            "gated_silu:": "gated-activation-half-v1; componentwise relative plus input-dependent subnormal allowance",
            "normalized_fanout:": "normalized-fanout-half-normwise-v1; every projection independently checked",
            "score:": "score-half-normwise-v1; original-input math.fsum QK transpose",
            "batched_fanout:": "grouped normalized projection; all branches and original-input math",
            "batched_rms:": "rms-half-normwise-v1; original-domain batched RMS and all device replicas",
            "input_attention_mixed:": "shared-gamma-input-attention-tail-half-normwise-v1; thirteen actual native observers and actual device branch gates",
            "attention_tail:": "supplied-qkv-attention-tail-half-normwise-v1; five actual observed branch gates",
            "prefill_tail:": "supplied-attention-tail-half-normwise-v1; separate actual native/device projection and delta gates",
            "feed_forward:": "normalized-feed-forward-half-normwise-v1; separate actual native/device delta gates plus original-input final math",
            "projection_residual_rms:": "projection-residual-rms-half-normwise-v1; original-input fsum projection/add/RMS",
            "mlp_blocked:": "rectangular-mlp-half-normwise-v1; explicit half block/f32 merge; original-input fsum/exp",
            "mlp:": "rectangular-mlp-half-normwise-v1; original-input fsum/exp three-projection gated path",
            "attention:": "attention-half-normwise-v1; original-input fsum/exp/fsum full supplied-QKV path",
            "score_softmax:": "score-softmax-half-normwise-v1; original-input dots and normalization, row mass and nonnegativity",
            "device_matmul:": "device-matmul-half-normwise-v1; original-input math.fsum contraction",
            "normalized_matmul:": "normalized-matmul-half-normwise-v1; standard composite numerical accuracy",
            "distributed_softmax:": "softmax-half-normwise-v1; row mass and nonnegativity, not per-component accuracy",
            "distributed_rms:": "rms-half-normwise-v1; not f32 or per-component accuracy",
        },
        "sources": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "tests/support/fixtures.py",
                "tests/support/solver_fixtures.py",
                "tests/support/pcg_fixtures.py",
                "tests/support/bicgstab_fixtures.py",
                "tests/support/power_fixtures.py",
                "tests/support/half_fixtures.py",
                "tests/support/grouped_half_fixtures.py",
                "tests/support/fft_fixtures.py",
                "tests/support/rms_fixtures.py",
                "tests/support/softmax_fixtures.py",
                "tests/support/normalized_matmul_fixtures.py",
                "tests/support/normalized_fanout_fixtures.py",
                "tests/support/score_fixtures.py",
                "tests/support/device_matmul_fixtures.py",
                "tests/support/score_softmax_fixtures.py",
                "tests/support/attention_fixtures.py",
                "tests/support/mlp_fixtures.py",
                "tests/support/projection_residual_rms_fixtures.py",
                "tests/support/feed_forward_fixtures.py",
                "tests/support/batched_rms_fixtures.py",
                "tests/support/batched_fanout_fixtures.py",
                "tests/support/batched_ffn_fixtures.py",
                "tests/support/cache_attention_fixtures.py",
                "tests/support/projected_cache_fixtures.py",
                "tests/support/composed_ffn_fixtures.py",
                "experiments/batched_ffn_gate.py",
                "tests/support/rms_fixtures.py",
                "tests/support/input_attention_fixtures.py",
                "experiments/input_attention_mixed_gate.py",
                "tests/support/attention_tail_fixtures.py",
                "experiments/attention_tail_gate.py",
                "tests/support/prefill_tail_fixtures.py",
                "experiments/prefill_tail_gate.py",
                "experiments/feed_forward_gate.py",
                "tests/support/swiglu_fixtures.py",
                "tests/support/pair_rotation_fixtures.py",
                "lib/Numerics/binary16.py",
                "lib/Numerics/roundoff.py",
                "scripts/authoring/import_stencil.py",
                "benchmarks/catalog.json",
                "tools/run_profiles.py",
            )
        },
    }
    summary = ROOT / "build/reports" / (run + ".json")
    summary.parent.mkdir(parents=True, exist_ok=True)

    def save():
        summary.write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(run, flush=True)
    if a.sdk:
        sif = Path(
            os.environ["PRAGMA_SDK_IMAGE"]
        )
        h = hashlib.sha256()
        with sif.open("rb") as f:
            for data in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(data)
        if (
            h.hexdigest()
            != "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"
        ):
            raise ValueError("SDK hash")
        report["sdk_sha256"] = h.hexdigest()
        save()
    for item in selected:
        key = item["project"] + "/" + item["kernel"]
        d = ROOT / json.loads((ROOT / "hls-layout.json").read_text())["profiles"][key]
        run_dir = ROOT / "build/runs" / key / run
        case = {
            "key": key,
            "passed": False,
            "artifact": str((run_dir).relative_to(ROOT)),
        }
        report["cases"].append(case)
        save()
        try:
            inputs = batches(
                item["fixture"],
                count={
                    "grid": 2,
                    "mesh_cg": 8,
                    "mesh_pcg": 9,
                    "mesh_bicgstab": 10,
                    "mesh_power": 8,
                    "half_gemm": 6,
                    "grouped_half": 8,
                    "distributed_fft": 6,
                    "distributed_rms": 6,
                    "distributed_softmax": 6,
                    "normalized_matmul": 6,
                    "normalized_fanout": 6,
                    "score": 6,
                    "device_matmul": 6,
                    "score_softmax": 6,
                    "attention": 6,
                    "mlp": 6,
                    "mlp_blocked": 8,
                    "projection_residual_rms": 8,
                    "feed_forward": 8,
                    "batched_rms": 8,
                    "batched_fanout": 8,
                    "batched_ffn": 8,
                    "cache_attention": 8,
                    "projected_cache": 8,
                    "composed_ffn": 8,
                    "input_attention_mixed": 8,
                    "attention_tail": 8,
                    "prefill_tail": 8,
                    "gated_silu": 6,
                    "pair_rotation": 6,
                }.get(item["fixture"].split(":")[0], 4),
            )
            out = build(
                d / "hls.cpp",
                run_dir,
                epochs=len(inputs),
                bound={
                    "distributed_rms": 1,
                    "distributed_softmax": 1024,
                    "normalized_matmul": 1,
                    "normalized_fanout": 1,
                    "score": 1,
                    "device_matmul": 1,
                    "score_softmax": 1,
                    "attention": 1,
                    "mlp": 1,
                    "mlp_blocked": 1,
                    "projection_residual_rms": 2,
                    "feed_forward": 2,
                    "batched_rms": 2,
                    "batched_fanout": 2,
                    "batched_ffn": 2,
                    "cache_attention": 2,
                    "projected_cache": 2,
                    "composed_ffn": 2,
                    "input_attention_mixed": 2,
                    "attention_tail": 2,
                    "prefill_tail": 2,
                    "gated_silu": 8,
                    "pair_rotation": 8,
                }.get(item["fixture"].split(":")[0], 64),
                batches=inputs,
                partitions=item["partitions"],
                sdk_options=report["sdk_runtime_options"],
                instrumentation=a.instrumentation or item.get("instrumentation"),
            )
            expected = json.loads((out / "reference.json").read_text())["outputs"]
            case["ir_application_checks"] = [
                check_application(item["fixture"], b, o)
                for b, o in zip(inputs, expected)
            ]
            native = parse_outputs((out / "native-output.txt").read_text())
            if len(native) != len(inputs):
                raise ValueError("native application epoch count")
            case["native_application_checks"] = [
                check_application(item["fixture"], b, o) for b, o in zip(inputs, native)
            ]
            if item["fixture"].startswith("composed_ffn:"):
                sys.path.insert(0, str(ROOT / "experiments"))
                from composed_ffn_gate import seal_native, seal_target

                case["native_application_checks"] = seal_native(out)["checks"]
                case["predicted_target_application_checks"] = seal_target(out)["checks"]
            if item["fixture"].startswith("projected_cache:"):
                sys.path.insert(0, str(ROOT / "experiments"))
                from projected_cache_gate import seal_native, seal_target

                case["native_application_checks"] = seal_native(out)["checks"]
                case["predicted_target_application_checks"] = seal_target(out)["checks"]
            if item["fixture"].startswith("cache_attention:"):
                sys.path.insert(0, str(ROOT / "experiments"))
                from cache_attention_gate import seal_native, seal_target

                case["native_application_checks"] = seal_native(out)["checks"]
                case["predicted_target_application_checks"] = seal_target(out)["checks"]
            if item["fixture"].startswith("batched_ffn:"):
                sys.path.insert(0, str(ROOT / "experiments"))
                from batched_ffn_gate import seal_native, seal_target

                case["native_application_checks"] = seal_native(out)["checks"]
                case["predicted_target_application_checks"] = seal_target(out)["checks"]
            if item["fixture"].startswith("batched_fanout:"):
                sys.path.insert(0, str(ROOT / "experiments"))
                from application_gate import seal

                dims = list(map(int, item["fixture"].split(":")[1:]))
                seal(
                    out,
                    ROOT / "tests/support/batched_fanout_fixtures.py",
                    lambda b, o: check_application(item["fixture"], b, o),
                    dict(zip(("B", "N", "F", "C"), dims)),
                )
            if item["fixture"].startswith("batched_rms:"):
                sys.path.insert(0, str(ROOT / "experiments"))
                from application_gate import seal

                dims = list(map(int, item["fixture"].split(":")[1:]))
                seal(
                    out,
                    ROOT / "tests/support/rms_fixtures.py",
                    lambda b, o: check_application(item["fixture"], b, o),
                    dict(B=dims[0], N=dims[1]),
                )
            if item["fixture"].startswith("input_attention_mixed:"):
                sys.path.insert(0, str(ROOT / "experiments"))
                from input_attention_mixed_gate import seal_native, seal_replay

                case["native_application_checks"] = seal_native(out)["checks"]
                case["target_replay_application_checks"] = seal_replay(
                    out, ROOT / item["target_replay"]
                )["checks"]
            if item["fixture"].startswith("attention_tail:"):
                sys.path.insert(0, str(ROOT / "experiments"))
                from attention_tail_gate import seal_native, seal_target

                case["native_application_checks"] = seal_native(out)["checks"]
                case["predicted_target_application_checks"] = seal_target(out)["checks"]
            if item["fixture"].startswith("prefill_tail:"):
                sys.path.insert(0, str(ROOT / "experiments"))
                from prefill_tail_gate import seal_native, seal_target

                gate = seal_native(out)
                case["native_application_checks"] = gate["checks"]
                seal_target(out)
            if item["fixture"].startswith("feed_forward:"):
                sys.path.insert(0, str(ROOT / "experiments"))
                from feed_forward_gate import seal_native, seal_target

                gate = seal_native(out)
                case["native_application_checks"] = gate["checks"]
                case["predicted_target_application_checks"] = seal_target(out)["checks"]
            case["native_application_source"] = (
                "native-output.txt: actual C++ executable stdout"
            )
            case["numerical_validation"] = numerical_summary(case)
            save()
            if a.sdk:
                env = dict(
                    os.environ,
                    SINGULARITYENV_CS_TARGET="SDR",
                    SINGULARITYENV_PYTHONUNBUFFERED="1",
                )
                with (out / "sdk.log").open("w") as log:
                    run_sdk(
                        [
                            os.environ["PRAGMA_CS_PYTHON"],
                            str(out / "implementation/sdk.py"),
                            str(out),
                        ],
                        out,
                        env,
                        log,
                        a.sdk_timeout,
                    )
                case["audit"] = audit(out)
                actual = json.loads((out / "results.json").read_text())["cases"]
                case["device_application_checks"] = [
                    check_application(item["fixture"], b, o)
                    for b, o in zip(inputs, actual)
                ]
                if item["fixture"].startswith("composed_ffn:"):
                    from composed_ffn_gate import device_checks

                    case["device_application_checks"] = device_checks(out)
                if item["fixture"].startswith("projected_cache:"):
                    from projected_cache_gate import device_checks

                    case["device_application_checks"] = device_checks(out)
                if item["fixture"].startswith("cache_attention:"):
                    from cache_attention_gate import device_checks

                    case["device_application_checks"] = device_checks(out)
                if item["fixture"].startswith("feed_forward:"):
                    from feed_forward_fixtures import check as ff_check
                    from mesh_common import unpack_tiles

                    sch = json.loads((out / "schedule.json").read_text())
                    observed = json.loads((out / "results.json").read_text())[
                        "diagnostics"
                    ]
                    case["device_application_checks"] = [
                        ff_check(
                            sch["M"],
                            sch["N"],
                            sch["F"],
                            sch["epsilon"],
                            b,
                            o,
                            unpack_tiles(
                                np.asarray(v["down_snapshot"], np.uint16).view(
                                    np.float16
                                ),
                                sch["Mt"],
                                sch["Nt"],
                                "F",
                            ),
                        )
                        for b, o, v in zip(inputs, actual, observed)
                    ]
                if item["fixture"].startswith("input_attention_mixed:"):
                    from input_attention_mixed_gate import device_checks

                    case["device_application_checks"] = device_checks(out)
                if item["fixture"].startswith("attention_tail:"):
                    from attention_tail_gate import device_checks

                    case["device_application_checks"] = device_checks(out)
                if item["fixture"].startswith("prefill_tail:"):
                    from prefill_tail_fixtures import check as tail_check
                    from mesh_common import unpack_tiles

                    sch = json.loads((out / "schedule.json").read_text())
                    observed = json.loads((out / "results.json").read_text())[
                        "diagnostics"
                    ]

                    def decode(v, key):
                        return unpack_tiles(
                            np.asarray(v[key], np.uint16).view(np.float16),
                            sch["Mt"],
                            sch["Nt"],
                            "F",
                        )

                    case["device_application_checks"] = [
                        tail_check(
                            sch["M"],
                            sch["N"],
                            sch["F"],
                            sch["epsilon"],
                            b,
                            o,
                            decode(v, "projection_snapshot"),
                            decode(v, "down_snapshot"),
                        )
                        for b, o, v in zip(inputs, actual, observed)
                    ]
                case["numerical_validation"] = numerical_summary(case)
            case["passed"] = True
            case["level"] = "sdk_simulator" if a.sdk else "native_cpu"
            save()
            print(key, "PASS", flush=True)
        except Exception as e:
            case["error"] = str(e)
            save()
            print(key, "FAIL", str(e), flush=True)
    report["success"] = bool(report["cases"]) and all(
        c["passed"] for c in report["cases"]
    )
    save()
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
