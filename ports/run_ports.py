"""Per-kernel fresh builds, staged artifacts and independent application checks."""

import argparse, datetime, json, subprocess, sys, os, hashlib, platform
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "toolchain"))
from compile import build
from validate import audit
from sdk_process import run_sdk
from fixtures import batches, check_application
from native_transport import parse_outputs


def numerical_summary(case):
    native = case.get("native_application_checks", [])
    device = case.get("device_application_checks", [])
    normwise_limits = {
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
        for item in json.loads((ROOT / "catalog.json").read_text())
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
            "mesh_gemm:": "componentwise-f32-dot-v1; fixed accuracy reported separately",
            "distributed_fft:": "fft-f32-normwise-v1; not per-component accuracy",
            "pair_rotation:": "pair-rotation-half-v1; cancellation-aware product-magnitude criterion",
            "gated_silu:": "gated-activation-half-v1; componentwise relative plus input-dependent subnormal allowance",
            "normalized_fanout:": "normalized-fanout-half-normwise-v1; every projection independently checked",
            "score:": "score-half-normwise-v1; original-input math.fsum QK transpose",
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
                "fixtures.py",
                "solver_fixtures.py",
                "pcg_fixtures.py",
                "bicgstab_fixtures.py",
                "power_fixtures.py",
                "half_fixtures.py",
                "grouped_half_fixtures.py",
                "fft_fixtures.py",
                "rms_fixtures.py",
                "softmax_fixtures.py",
                "normalized_matmul_fixtures.py",
                "normalized_fanout_fixtures.py",
                "score_fixtures.py",
                "device_matmul_fixtures.py",
                "score_softmax_fixtures.py",
                "attention_fixtures.py",
                "mlp_fixtures.py",
                "projection_residual_rms_fixtures.py",
                "swiglu_fixtures.py",
                "pair_rotation_fixtures.py",
                "toolchain/binary16.py",
                "toolchain/roundoff.py",
                "import_stencil.py",
                "catalog.json",
                "run_ports.py",
            )
        },
    }
    summary = ROOT / "evidence" / (run + ".json")

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
        d = ROOT / "projects" / key
        case = {
            "key": key,
            "passed": False,
            "artifact": str((d / run).relative_to(ROOT)),
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
                    "gated_silu": 6,
                    "pair_rotation": 6,
                }.get(item["fixture"].split(":")[0], 4),
            )
            out = build(
                d / "hls.cpp",
                d / run,
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
