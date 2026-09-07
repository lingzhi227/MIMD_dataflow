"""Build rectangular gated MLP through the shared HLS pipeline."""

import argparse, datetime, json, shutil, sys, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]
from compile import build
from mlp_fixtures import batches, check
from native_transport import parse_outputs


def source(m, n, f, p, blocked=False):
    proj = f"#pragma csl dataflow rows={p} cols={p} exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed"
    gate = f"#pragma csl dataflow rows={p} cols={p} partition=tiles math=sdk_half compute=dsr elementwise=map fp=relaxed"
    down_policy = proj + (" accumulation=block_f32" if blocked else "")
    down_call = (
        f"spatial::matmul_blocked<spatial::f16,spatial::scalar>(hidden,d,{f//p})"
        if blocked
        else "spatial::matmul(hidden,d)"
    )
    return f"""#include "spatial.hpp"
void design() {{
 auto x=spatial::input<{m},{n},spatial::f16>("x",0.125);
 auto u=spatial::input<{n},{f},spatial::f16>("up_weight",0.125);
 auto g=spatial::input<{n},{f},spatial::f16>("gate_weight",0.125);
 auto d=spatial::input<{f},{n},spatial::f16>("down_weight",0.125);
 {proj}
 auto up=spatial::matmul(x,u);
 {proj}
 auto gate=spatial::matmul(x,g);
 {gate}
 auto activated=spatial::silu(gate);
 {gate}
 auto hidden=spatial::multiply(up,activated);
 {down_policy}
 auto result={down_call};
 spatial::output("output",result);
}}
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--f", type=int, default=256)
    p.add_argument("--p", type=int, default=8)
    p.add_argument(
        "--instrumentation", choices=["sampled", "counters"], default="sampled"
    )
    p.add_argument("--blocked-down", action="store_true")
    a = p.parse_args()
    folder = (
        ROOT
        / "projects/waferllm"
        / (
            f"mlp_{a.m}x{a.n}x{a.f}_{a.p}x{a.p}"
            + ("_blocked" if a.blocked_down else "")
        )
    )
    folder.mkdir(exist_ok=True)
    src = source(a.m, a.n, a.f, a.p, a.blocked_down)
    if (folder / "hls.cpp").exists():
        assert (folder / "hls.cpp").read_text() == src
    else:
        (folder / "hls.cpp").write_text(src)
        (folder / "PORT.json").write_text(
            json.dumps(
                dict(
                    project="waferllm",
                    kernel=folder.name,
                    origins=[
                        "Prefill/src/prefill.csl z1_matmul/z2_matmul/z3_comp/h2_matmul"
                    ],
                    source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                    status="development",
                    partitions=1,
                    contract="Resident rectangular supplied X/U/G/D gated MLP; no RMS/residual/full-model claim.",
                ),
                indent=2,
            )
            + "\n"
        )
    dest = folder / (
        "run-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    workload = batches(a.m, a.n, a.f, a.p if a.blocked_down else None)
    build(
        folder / "hls.cpp",
        dest,
        epochs=len(workload),
        bound=1,
        batches=workload,
        instrumentation=a.instrumentation,
        sdk_options=dict(suppress_trace=True, num_threads=8, dump_core=True),
    )
    # Application math is deliberately outside compiler/codegen. Seal the actual
    # native stdout check before handing the fresh bundle to the SDK driver.
    checks = []
    failure = None
    bs = json.loads((dest / "batches.json").read_text())
    native = parse_outputs((dest / "native-output.txt").read_text())
    assert len(bs) == len(native) == len(workload)
    for epoch, (b, o) in enumerate(zip(bs, native)):
        try:
            checks.append(check(a.m, a.n, a.f, b, o))
        except AssertionError as error:
            failure = dict(epoch=epoch, error=repr(error))
            break
    reference = dest / "application-reference.py"
    shutil.copyfile(ROOT / "mlp_fixtures.py", reference)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    gate = dict(
        passed=failure is None,
        dimensions=dict(M=a.m, N=a.n, F=a.f),
        checks=checks,
        failure=failure,
        native_output_sha256=sha(dest / "native-output.txt"),
        reference_sha256=sha(reference),
        scope="actual C++ stdout checked against original-input independent fsum/exp; required before SDK",
    )
    (dest / "application-gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    manifest = json.loads((dest / "manifest.json").read_text())
    manifest["application_gate"] = "application-gate.json"
    for name in ("application-gate.json", "application-reference.py"):
        manifest["files"][name] = sha(dest / name)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not gate["passed"]:
        (dest / "stage.json").write_text(
            json.dumps(
                dict(
                    stage="native_application_failed",
                    epoch=failure["epoch"],
                    compiler_native_equivalence_passed=True,
                    sdk_started=False,
                    application_gate="application-gate.json",
                )
            )
            + "\n"
        )
    assert gate[
        "passed"
    ], f"native application accuracy failed at epoch {failure['epoch']}; SDK must not run"
    shutil.copyfile(
        ROOT / "experiments/execute_frozen_bundle.py", dest / "sdk-execution-driver.py"
    )
    print(dest.relative_to(ROOT), flush=True)
