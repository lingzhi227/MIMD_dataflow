"""Reproducible middleware build: AST -> checked IR -> schedule -> CSL/manifest."""

import argparse, hashlib, json, subprocess, shutil, sys
from pathlib import Path
from frontend import parse, check
from ir import verify, evaluate
from planner import optimize, plan
from backend import generate
from float32 import close
from native_transport import parse_outputs

ROOT = Path(__file__).resolve().parent


def build(
    source,
    dest,
    epochs=4,
    bound=8,
    identity=False,
    batches=None,
    partitions=1,
    sdk_options=None,
    instrumentation=None,
):
    source = Path(source).resolve()
    dest = Path(dest).resolve()
    check(not dest.exists(), "output must be fresh")
    dest.mkdir(parents=True)

    def stage(name):
        (dest / "stage.json").write_text(json.dumps({"stage": name}) + "\n")

    (dest / "source.cpp").write_text(source.read_text())
    stage("frontend")
    raw = parse(source, dest)
    (dest / "01_frontend_ir.json").write_text(json.dumps(raw, indent=2) + "\n")
    (dest / "build-configuration.json").write_text(
        json.dumps(
            dict(
                epochs=epochs,
                input_bound=bound,
                partitions=partitions,
                identity=identity,
                instrumentation=instrumentation,
                sdk_options=sdk_options,
            ),
            indent=2,
        )
        + "\n"
    )
    stage("semantic_verification")
    verification_input = dict(raw)
    # Observation storage affects admissible shapes. Resolve the requested mode
    # before any verifier invokes its resource planner; never budget a different
    # default mode and only apply the user's mode afterward.
    if instrumentation is not None:
        check(
            instrumentation in ("sampled", "counters"), "unknown instrumentation mode"
        )
        verification_input["instrumentation"] = instrumentation
    original = verify(verification_input, epochs, bound)
    if instrumentation is not None:
        check(
            original.get("profile")
            in (
                "mesh_pair_rotation.v1",
                "mesh_score.v1",
                "mesh_device_matmul.v1",
                "mesh_score_softmax.v1",
                "mesh_attention.v1",
                "mesh_swiglu.v1",
                "mesh_normalized_fanout.v1",
                "mesh_normalized_matmul.v1",
                "mesh_softmax.v1",
                "mesh_rms.v1",
                "mesh_fft.v1",
                "mesh_qr.v1",
                "mesh_twohop.v1",
                "mesh_grouped_gemv.v1",
            )
            and instrumentation in ("sampled", "counters"),
            "explicit instrumentation modes require QR/two-hop/grouped sampled/counters",
        )
        original["instrumentation"] = instrumentation
    if sdk_options is not None:
        from mesh_common import validate_sdk_options

        validate_sdk_options(sdk_options)
        check(
            original.get("profile")
            in (
                "mesh_pair_rotation.v1",
                "mesh_score.v1",
                "mesh_device_matmul.v1",
                "mesh_score_softmax.v1",
                "mesh_attention.v1",
                "mesh_swiglu.v1",
                "mesh_normalized_fanout.v1",
                "mesh_normalized_matmul.v1",
                "mesh_softmax.v1",
                "mesh_rms.v1",
                "mesh_fft.v1",
                "mesh_gemv.v1",
                "mesh_gemm.v1",
                "mesh_cannon.v1",
                "mesh_twohop.v1",
                "mesh_grouped_gemv.v1",
                "mesh_cholesky.v1",
                "mesh_lu.v1",
                "mesh_qr.v1",
                "mesh_spmv.v1",
                "mesh_cg.v1",
                "mesh_power.v1",
                "mesh_reduction.v1",
            ),
            "explicit SDK trace options require a supported mesh profile",
        )
        (dest / "runtime-options.json").write_text(
            json.dumps(sdk_options, indent=2) + "\n"
        )
    (dest / "02_checked_ir.json").write_text(json.dumps(original, indent=2) + "\n")
    stage("optimization")
    module = optimize(original, identity)
    (dest / "03_optimized_ir.json").write_text(json.dumps(module, indent=2) + "\n")
    stage("spatial_planning")
    schedule = plan(module, partitions)
    # Input generation is a harness convenience, not part of synthesized arithmetic.
    if batches is None:
        batches = [
            {
                n["host"]: [
                    ((i * 3 + e * 5) % (2 * bound + 1)) - bound
                    for i in range(n["shape"][0] * n["shape"][1])
                ]
                for n in module["nodes"]
                if n["op"] == "input"
            }
            for e in range(epochs)
        ]
    expected, history = evaluate(module, batches)
    check(
        module.get("profile")
        in (
            "grid.v1",
            "mesh_pair_rotation.v1",
            "mesh_score.v1",
            "mesh_device_matmul.v1",
            "mesh_score_softmax.v1",
            "mesh_attention.v1",
            "mesh_swiglu.v1",
            "mesh_normalized_fanout.v1",
            "mesh_normalized_matmul.v1",
            "mesh_softmax.v1",
            "mesh_rms.v1",
            "mesh_fft.v1",
            "mesh_gemv.v1",
            "mesh_gemm.v1",
            "mesh_cannon.v1",
            "mesh_twohop.v1",
            "mesh_grouped_gemv.v1",
            "mesh_cholesky.v1",
            "mesh_lu.v1",
            "mesh_qr.v1",
            "mesh_spmv.v1",
            "mesh_cg.v1",
            "mesh_power.v1",
            "mesh_reduction.v1",
        )
        or close(evaluate(dict(module, nodes=schedule["nodes"]), batches)[0], expected),
        "spatial lowering mismatch",
    )
    check(
        close(expected, evaluate(original, batches)[0]),
        "optimization semantic mismatch",
    )
    if module.get("profile") == "mesh_power.v1":
        from mesh_power import packing

        (dest / "sparse-packing.json").write_text(
            json.dumps(packing(module, batches), indent=2) + "\n"
        )
    if module.get("profile") == "mesh_cg.v1":
        from mesh_cg import packing

        (dest / "sparse-packing.json").write_text(
            json.dumps(packing(module, batches), indent=2) + "\n"
        )
    if module.get("profile") == "mesh_spmv.v1":
        from mesh_spmv import matrix
        from sparse_storage import partition, Capacity

        packing = [
            partition(
                matrix(module, b),
                schedule["rows"],
                schedule["cols"],
                Capacity(**schedule["capacity"]),
            )
            for b in batches
        ]
        (dest / "sparse-packing.json").write_text(json.dumps(packing, indent=2) + "\n")
    stage("code_generation")
    for name, value in [
        ("semantic", module),
        ("schedule", schedule),
        ("batches", batches),
        ("reference", {"outputs": expected, "history": history}),
    ]:
        (dest / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    (dest / "source.cpp").write_text(source.read_text())
    generate(schedule, dest)
    stage("native_validation")
    cmd = [
        "clang++",
        "-std=c++17",
        "-ffp-contract=off",
        "-DMW_BOUND=" + str(bound),
        "-DMW_EPOCHS=" + str(epochs),
        "-DMW_MAX_INPUT="
        + str(
            max(
                n["shape"][0] * n["shape"][1]
                for n in module["nodes"]
                if n["op"] in ("input", "index_input")
            )
        ),
        "-Werror",
        "-Wno-unknown-pragmas",
        "-fsanitize=undefined",
        "-fno-sanitize-recover=all",
        "-I",
        str(ROOT / "include"),
        str(source),
        str(ROOT / "runtime/native.cpp"),
        "-o",
        str(dest / "native"),
    ]
    (dest / "native-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    compiled = subprocess.run(cmd, text=True, capture_output=True)
    (dest / "native-compile.log").write_text(compiled.stdout + compiled.stderr)
    check(compiled.returncode == 0, "native compilation failed; see native-compile.log")
    data = str(epochs) + "\n"
    for b in batches:
        data += str(len(b)) + "\n"
        for name, v in b.items():
            index = any(
                n["op"] == "index_input" and n["host"] == name for n in module["nodes"]
            )
            data += (
                ("@u32 " if index else "")
                + name
                + " "
                + str(len(v))
                + " "
                + " ".join(map(str, v))
                + "\n"
            )
    (dest / "native-input.txt").write_text(data)
    result = subprocess.run(
        [str(dest / "native")], input=data, text=True, capture_output=True
    )
    (dest / "native-output.txt").write_text(result.stdout)
    (dest / "native-stderr.txt").write_text(result.stderr)
    check(result.returncode == 0, "native execution failed; see native-stderr.txt")
    got = parse_outputs(result.stdout)
    check(len(got) == len(expected), "native output epoch count")
    for actual, reference in zip(got, expected):
        for name, values in actual.items():
            if values and all(type(v) is int for v in values):
                check(values == reference.get(name), "native u32 output exactness")
    check(close(got, expected), "native C++ vs IR mismatch")
    if module.get("profile") in (
        "mesh_twohop.v1",
        "mesh_grouped_gemv.v1",
        "mesh_score.v1",
        "mesh_device_matmul.v1",
        "mesh_score_softmax.v1",
        "mesh_attention.v1",
    ):
        from binary16 import bits

        check(
            all(
                bits(x) == bits(y)
                for actual, reference in zip(got, expected)
                for key in actual
                for x, y in zip(actual[key], reference[key])
            ),
            "native binary16 fused reference bit agreement",
        )
    stage("native_passed_sdk_pending")
    files = (
        list(dest.glob("*.csl"))
        + list(dest.glob("*.json"))
        + list(dest.glob("*.txt"))
        + list(dest.glob("*.log"))
        + [dest / "source.cpp", dest / "native"]
    )
    impl = [
        ROOT / x
        for x in [
            "runtime/rms_layout.csl",
            "runtime/rms_pe.csl",
            "runtime/rms_row_reduce.csl",
            "mesh_score_softmax.py",
            "mesh_attention.py",
            "attention_debug.py",
            "mesh_attention_sdk.py",
            "runtime/attention_pe.csl",
            "runtime/attention_layout.csl",
            "score_softmax_debug.py",
            "mesh_score_softmax_sdk.py",
            "runtime/score_softmax_pe.csl",
            "runtime/score_softmax_layout.csl",
            "runtime/softmax_local.csl",
            "mesh_device_matmul.py",
            "device_matmul_debug.py",
            "mesh_device_matmul_sdk.py",
            "runtime/device_matmul_pe.csl",
            "runtime/device_matmul_layout.csl",
            "mesh_score.py",
            "score_debug.py",
            "mesh_score_sdk.py",
            "runtime/score_pe.csl",
            "runtime/score_layout.csl",
            "mesh_pair_rotation.py",
            "mesh_pair_rotation_sdk.py",
            "pair_rotation_debug.py",
            "include/pair_rotation.hpp",
            "runtime/pair_rotation_layout.csl",
            "runtime/pair_rotation_pe.csl",
            "mesh_swiglu.py",
            "swiglu_debug.py",
            "mesh_swiglu_sdk.py",
            "include/activation.hpp",
            "runtime/swiglu_pe.csl",
            "runtime/swiglu_layout.csl",
            "mesh_normalized_fanout.py",
            "mesh_normalized_fanout_sdk.py",
            "normalized_fanout_debug.py",
            "runtime/normalized_fanout_pe.csl",
            "runtime/normalized_fanout_layout.csl",
            "mesh_normalized_matmul.py",
            "normalized_matmul_debug.py",
            "mesh_normalized_matmul_sdk.py",
            "runtime/normalized_matmul_pe.csl",
            "runtime/normalized_matmul_layout.csl",
            "runtime/inference_comm.csl",
            "runtime/inference_routes.csl",
            "runtime/waferllm-LICENSE.txt",
            "mesh_softmax.py",
            "softmax_debug.py",
            "mesh_softmax_sdk.py",
            "runtime/softmax_pe.csl",
            "runtime/softmax_layout.csl",
            "runtime/row_chain.csl",
            "mesh_rms.py",
            "rms_debug.py",
            "include/normalization.hpp",
            "mesh_rms_sdk.py",
            "sdk_math_reference.py",
            "fft_debug.py",
            "native_transport.py",
            "mesh_fft.py",
            "mesh_fft_sdk.py",
            "compiler_parameters.py",
            "include/fft.hpp",
            "runtime/fft_layout.csl",
            "runtime/fft_pe.csl",
            "runtime/fft_driver.csl",
            "frontend.py",
            "pragma_contracts.py",
            "solver_ir.py",
            "solver_reference.py",
            "mesh_cg.py",
            "mesh_cg_sdk.py",
            "runtime/cg_controller.csl",
            "runtime/solver_common.csl",
            "runtime/bicgstab_controller.csl",
            "ir.py",
            "planner.py",
            "backend.py",
            "compile.py",
            "sdk.py",
            "include/spatial.hpp",
            "include/solver.hpp",
            "include/bicgstab.hpp",
            "include/power.hpp",
            "power_ir.py",
            "mesh_power.py",
            "resident_abi.py",
            "resident_sdk.py",
            "runtime/power_controller.csl",
            "bicgstab_reference.py",
            "bicgstab_sdk.py",
            "solver_audit.py",
            "power_audit.py",
            "runtime/native.cpp",
            "runtime/comm_runtime.csl",
            "integrity.py",
            "validate.py",
            "float32.py",
            "local_kernel.py",
            "debug.py",
            "roundoff.py",
            "mesh_common.py",
            "qr_schedule.py",
            "sparse_storage.py",
            "mesh_spmv.py",
            "mesh_reduction.py",
            "mesh_reduction_sdk.py",
            "runtime/reduction_layout.csl",
            "runtime/reduction_pe.csl",
            "runtime/scalar_allreduce.csl",
            "runtime/sdk_blas.csl",
            "mesh_spmv_sdk.py",
            "runtime/spmv_layout.csl",
            "runtime/spmv_kernel.csl",
            "runtime/spmv_pe.csl",
            "runtime/spmv_routes.csl",
            "runtime/u16_transport.csl",
            "runtime/spmv_reduce_pe.csl",
            "runtime/spmv_reduce_routes.csl",
            "mesh_qr.py",
            "mesh_qr_sdk.py",
            "runtime/mesh_qr_pe.csl",
            "runtime/mesh_qr_layout.csl",
            "mesh_lu.py",
            "mesh_lu_sdk.py",
            "runtime/mesh_lu_pe.csl",
            "runtime/mesh_lu_layout.csl",
            "factor_sdk.py",
            "mesh_cholesky.py",
            "mesh_cholesky_sdk.py",
            "runtime/mesh_cholesky_pe.csl",
            "runtime/mesh_cholesky_layout.csl",
            "runtime/mesh_cholesky_launch.csl",
            "mesh_gemm.py",
            "mesh_cannon.py",
            "mesh_twohop.py",
            "mesh_grouped_gemv.py",
            "grouped_gemv_sdk.py",
            "grouped_gemv_debug.py",
            "runtime/grouped_pe.csl",
            "runtime/grouped_layout.csl",
            "runtime/grouped_comm.csl",
            "runtime/grouped_routes.csl",
            "twohop_sdk.py",
            "twohop_debug.py",
            "binary16.py",
            "half_matrix.py",
            "runtime/twohop_pe.csl",
            "runtime/twohop_layout.csl",
            "runtime/twohop_comm.csl",
            "runtime/twohop_routes.csl",
            "runtime/cannon_pe.csl",
            "runtime/cannon_layout.csl",
            "mesh_gemm_sdk.py",
            "runtime/mesh_gemm_pe.csl",
            "runtime/mesh_gemm_layout.csl",
            "runtime/mesh_gemm_vector.csl",
            "mesh_gemv.py",
            "mesh_gemv_sdk.py",
            "runtime/mesh_gemv_pe.csl",
            "runtime/mesh_gemv_layout.csl",
            "grid_ir.py",
            "grid_plan.py",
            "grid_backend.py",
            "grid_sdk.py",
            "grid_validate.py",
            "sdk_process.py",
            "vectorize.py",
            "runtime/frame_runtime.csl",
        ]
    ]
    for path in impl:
        snapshot = dest / "implementation" / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot)
    manifest = {
        "implementation_snapshot": True,
        "version": "middleware.build.v1",
        "target": "SDK2.10.1/WSE3",
        "native_command": cmd,
        "frontend": raw["clang_version"],
        "source_sha256": raw["source_sha256"],
        "passes": ["identity"] if identity else [],
        "epochs": epochs,
        "expected_output_values": sum(len(v) for b in expected for v in b.values()),
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "implementation": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in impl
        },
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Resolve imports only from the frozen snapshot, before any SDK work is sent.
    subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/"implementation"));from integrity import verify_codegen;verify_codegen(p)',
            str(dest),
        ],
        check=True,
    )
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("-o", required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--bound", type=int, default=8)
    p.add_argument("--identity", action="store_true")
    p.add_argument("--partitions", type=int, default=1)
    p.add_argument("--batches", type=Path, help="JSON array of named input batches")
    p.add_argument("--instrumentation", choices=("sampled", "counters"))
    a = p.parse_args()
    print(
        build(
            a.source,
            a.o,
            a.epochs,
            a.bound,
            a.identity,
            batches=json.loads(a.batches.read_text()) if a.batches else None,
            partitions=a.partitions,
            instrumentation=a.instrumentation,
        )
    )
