"""Fresh resident Jacobi-PCG qualification run before catalog promotion."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import sys, json, datetime, os, hashlib
from pathlib import Path

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from compile import build
from pcg_fixtures import batches, check
from sdk_process import run_sdk
from validate import audit

run = "run-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
out = ROOT / "benchmarks/linear_algebra/sdk_examples/mesh_pcg_512_4x4" / run
report = dict(run=run, success=False, artifact=str(out.relative_to(ROOT)))
evidence = ROOT / "validation/evidence" / (run + ".json")


def save():
    evidence.write_text(json.dumps(report, indent=2) + "\n")


save()
print(run, flush=True)
try:
    sif = Path(
        "/home/lingzhi/cerebras/sdk/2.10.1/sdk-cbcore-2.10.1-sdk-202606181328-8faf87a26e.sif"
    )
    with sif.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    assert digest == "fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d"
    report["sdk_sha256"] = digest
    inputs = batches()
    build(
        out.parent / "hls.cpp",
        out,
        epochs=len(inputs),
        bound=64,
        batches=inputs,
        sdk_options=dict(suppress_trace=True, num_threads=16, dump_core=True),
    )
    report["native_checks"] = [
        check(b, o)
        for b, o in zip(
            inputs, json.loads((out / "reference.json").read_text())["outputs"]
        )
    ]
    save()
    with (out / "sdk.log").open("w") as log:
        run_sdk(
            [
                "/home/lingzhi/cerebras/sdk/2.10.1/cs_python",
                str(out / "implementation/sdk.py"),
                str(out),
            ],
            out,
            dict(
                os.environ,
                SINGULARITYENV_CS_TARGET="SDR",
                SINGULARITYENV_PYTHONUNBUFFERED="1",
            ),
            log,
            600,
        )
    report["audit"] = audit(out)
    report["device_checks"] = [
        check(b, o)
        for b, o in zip(inputs, json.loads((out / "results.json").read_text())["cases"])
    ]
    report["success"] = True
except BaseException as e:
    report["error"] = repr(e)
    raise
finally:
    save()
print("PASS", flush=True)
