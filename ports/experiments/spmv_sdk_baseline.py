"""Same-input SpMV comparison against the required WSE3 source migration.

The source-native path omits HLS partial witnesses/capture. It retains the
queue/UT/phase adaptations required on WSE3 and uses identical SDK timing.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_function(source, name):
    start = source.index("fn " + name + "(")
    pos = source.index("{", start) + 1
    depth = 1
    while depth:
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
        pos += 1
    return source[:start] + source[pos:]


def prepare(bundle, out):
    out.mkdir(exist_ok=False)
    for name in (
        "schedule.json",
        "semantic.json",
        "batches.json",
        "sparse-packing.json",
        "runtime-options.json",
    ):
        if (bundle / name).exists():
            shutil.copy2(bundle / name, out / name)
    shutil.copytree(bundle / "implementation", out / "implementation")
    for name in (
        "layout.csl",
        "kernel.csl",
        "spmv_pe.csl",
        "spmv_routes.csl",
        "u16_transport.csl",
    ):
        code = (bundle / name).read_text()
        if name == "kernel.csl":
            code = remove_function(code, "hls_capture")
        if name in ("kernel.csl", "layout.csl"):
            code = (
                "\n".join(line for line in code.splitlines() if "hls_" not in line)
                + "\n"
            )
        if name == "spmv_pe.csl":
            code = remove_function(code, "completion_counts")
            code = (
                "\n".join(
                    line for line in code.splitlines() if "partial_witness" not in line
                )
                + "\n"
            )
        (out / name).write_text(code)
    shutil.copy2(__file__, out / "driver.py")


def worker(root):
    sys.path.insert(0, str(root / "implementation"))
    from mesh_spmv_sdk import run

    run(root, diagnostics=False)


def main(bundle):
    sys.path.insert(0, str(ROOT / "toolchain"))
    from integrity import verify_bundle
    from sdk_process import run_sdk
    from mesh_spmv_sdk import check_result

    verify_bundle(bundle, implementation=False)
    s = read(bundle / "schedule.json")
    hls = read(bundle / "results.json")
    if s["profile"] != "mesh_spmv.v1" or not hls["success"]:
        raise ValueError("completed SpMV required")
    out = (
        ROOT
        / "evidence"
        / (
            "native-spmv-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    prepare(bundle, out)
    sif = Path(
        "/home/lingzhi/cerebras/sdk/2.10.1/sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif"
    )
    image_sha = sha(sif)
    if image_sha != "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d":
        raise ValueError("SDK image")
    provenance = dict(
        sdk_sha256=image_sha,
        kind="SDK hypersparse source with required WSE3 queue/UT/phase adaptation, no partial-sample arithmetic or capture; local timing only",
        upstream_commit="4866cf330333446cb5e529e10f36be4600d1df29",
        comparison_bundle=str(bundle),
        comparison_results_sha256=sha(bundle / "results.json"),
        files={str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    (out / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(out, flush=True)
    with (out / "sdk.log").open("w") as log:
        run_sdk(
            [
                "/home/lingzhi/cerebras/sdk/2.10.1/cs_python",
                str(out / "driver.py"),
                "--worker",
                str(out),
            ],
            out,
            dict(
                os.environ,
                SINGULARITYENV_CS_TARGET="SDR",
                SINGULARITYENV_PYTHONUNBUFFERED="1",
            ),
            log,
            900,
        )
    native = read(out / "results.json")
    m = read(out / "semantic.json")
    batches = read(out / "batches.json")
    if not native["success"] or len(native["cases"]) != len(batches):
        raise ValueError("native lifecycle")
    numerical = []
    for batch, result in zip(batches, native["cases"]):
        v, r, p, x = [batch[n["host"]] for n in m["nodes"][:4]]
        numerical.append(
            check_result(s["M"], s["N"], p, r, v, x, result[m["nodes"][-1]["host"]])
        )

    def times(result):
        return [
            [
                (
                    sum(int(w[i + 3]) << (16 * i) for i in range(3))
                    - sum(int(w[i]) << (16 * i) for i in range(3))
                )
                % (1 << 48)
                for row in d["timing"]
                for w in row
            ]
            for d in result["diagnostics"]
        ]

    nt, ht = times(native), times(hls)
    if any(not 0 < t < 1 << 32 for epoch in nt + ht for t in epoch):
        raise ValueError("timestamp bounds")
    report = dict(
        passed=True,
        epochs=len(batches),
        numerical_checks=numerical,
        native_per_pe_cycles=nt,
        hls_per_pe_cycles=ht,
        max_local_ratios=[max(h) / max(n) for h, n in zip(ht, nt)],
        outputs_exact_equal=native["cases"] == hls["cases"],
        scope="Maximum local SDK tic/toc interval including command launch; same four input epochs and runtime options. Excludes host I/O; not synchronized global latency or hardware measurement.",
        native_results_sha256=sha(out / "results.json"),
    )
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("passed", "max_local_ratios", "outputs_exact_equal")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    worker(args.bundle.resolve()) if args.worker else main(args.bundle.resolve())
