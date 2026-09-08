"""Same full graph; unchanged pinned Decode local vecmat bodies for nine contractions."""

import argparse, datetime, json, shutil, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(
    0, str(HERE if (HERE / "probe_runtime.py").exists() else ROOT / "experiments")
)
from probe_runtime import verify, execute, mesh_half_worker, read, sha


def function(text, name):
    start = text.index("fn " + name + "(")
    brace = text.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1
    return text[start:end]


def prepare(bundle):
    bundle = bundle.resolve()
    verify(bundle)
    assert read(bundle / "stage.json")["stage"] == "ready_for_sdk"
    root = (
        ROOT
        / "evidence"
        / (
            "hls-composed-source-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    for p in bundle.glob("*.csl"):
        shutil.copy2(p, root / p.name)
    for name in (
        "inputs.json",
        "expected.json",
        "schema.json",
        "runtime-options.json",
        "sdk-command.json",
        "probe_runtime.py",
        "sdk_process.py",
        "schedule.json",
    ):
        shutil.copy2(bundle / name, root / name)
    previous = ROOT / "evidence/projected-cache-source-compute-20260908T000650498000Z"
    verify(previous)
    upstream = ROOT / "projects/waferllm/upstream/Decode/src/decode.csl"
    code = (previous / "source_vecmat.csl").read_text()
    source = upstream.read_text()
    bodies = {}
    for name in ("gemv_static_step", "vecmat_computation"):
        assert function(code, name) == function(source, name)
        import hashlib

        bodies[name] = hashlib.sha256(function(source, name).encode()).hexdigest()
    (root / "source_vecmat.csl").write_text(code)
    p = root / "batched_matmul_blocked.csl"
    text = p.read_text()
    assert text.count('"batched_matmul_local.csl"') == 2
    p.write_text(text.replace('"batched_matmul_local.csl"', '"source_vecmat.csl"'))
    shutil.copy2(__file__, root / "driver.py")
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                scope="Nine local contractions use unchanged pinned Decode gemv_static_step/vecmat_computation, wrapper initialization included in timer. RMS/pairs/softmax/SiLU/SDK communication are shared. Not original full Decode execution or whole-source performance.",
                upstream_sha256=sha(upstream),
                original_functions_sha256=bodies,
                hls_provenance_sha256=sha(bundle / "provenance.json"),
                files={p.name: sha(p) for p in root.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT), flush=True)


def run(root):
    root = root.resolve()
    execute(root, 21600)
    actual = read(root / "results.json")
    expected = read(root / "expected.json")
    assert actual["success"] and len(actual["cases"]) == len(expected) == 8
    for e, (case, want) in enumerate(zip(actual["cases"], expected)):
        for k, v in want.items():
            np.testing.assert_array_equal(
                case[k], v, err_msg=f"{e} source actual raw {k}"
            )
        assert np.all((np.asarray(case["queues"]) & 60) == 60)
    (root / "source-raw-review.json").write_text(
        json.dumps(
            dict(
                passed=True,
                epochs=8,
                results_sha256=sha(root / "results.json"),
                provenance_sha256=sha(root / "provenance.json"),
            ),
            indent=2,
        )
        + "\n"
    )
    print("SOURCE COMPUTE FULL RAW PASS", root, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", type=Path)
    p.add_argument("--execute", type=Path)
    p.add_argument("--worker", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.prepare)
    elif a.worker:
        mesh_half_worker(a.worker.resolve())
    elif a.execute:
        run(a.execute)
