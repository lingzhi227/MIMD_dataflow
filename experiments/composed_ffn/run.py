"""Fresh full HLS frontend/native/CSL/SDK experiments; no catalog auto-admission."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, shutil, subprocess, sys, traceback
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
HERE = Path(__file__).resolve().parent
sys.path[:0] = (
    [str(HERE / "implementation"), str(HERE)]
    if (HERE / "implementation").exists()
    else [str(ROOT / "lib"), str(ROOT / "experiments"), str(ROOT)]
)
from probe_runtime import execute, mesh_half_worker, read, sha


def metric(actual, ideal):
    e = np.asarray(actual) - ideal
    return dict(
        relative_l2=float(np.linalg.norm(e) / max(np.linalg.norm(ideal), 1e-12)),
        relative_peak=float(np.max(np.abs(e)) / max(np.max(np.abs(ideal)), 1e-12)),
    )


def roles(s):
    a = s["graph"]["attention"]["nodes"]
    t = s["graph"]["ffn"]["nodes"]
    return {
        n["id"]: name
        for n, name in zip(
            a[10:22],
            (
                "normalized",
                "query",
                "key_projection",
                "value_projection",
                "rotated_query",
                "rotated_key",
                "transpose",
                "score",
                "probability",
                "context",
                "delta",
                "result",
            ),
        )
        if name != "transpose"
    } | {
        n["id"]: name
        for n, name in zip(
            t[5:12],
            (
                "ffn_normalized",
                "up",
                "gate",
                "activation",
                "hidden",
                "ffn_delta",
                "final_result",
            ),
        )
    }


def fixtures():
    from projected_cache_fixtures import batches

    rows = batches(3, 256, 512)
    for e, row in enumerate(rows):
        for name, t, den, shape in [
            ("wu", 0, 512, (256, 512)),
            ("wg", 1, 512, (256, 512)),
            ("wd", 2, 4096, (512, 256)),
        ]:
            row[name] = [
                (((i * 13 + j * 5 + e * 7 + t * 11) % 33) - 16) / den
                for i in range(shape[0])
                for j in range(shape[1])
            ]
    # A legal full-graph overflow witness, not a post-attention injected input.
    row = rows[7]
    for k in ("x", "gamma", "key", "value", "cosine"):
        row[k] = [1.0] * len(row[k])
    row["sine"] = [0.0] * len(row["sine"])
    for k in ("wq", "wk", "wv", "wu", "wg"):
        row[k] = [1 / 32] * len(row[k])
    row["wo"] = [1 / 8] * len(row["wo"])
    row["wd"] = [1 / 256] * len(row["wd"])
    return rows


def prepare():
    from frontend import parse
    from projected_cache_ffn_plan import plan
    from projected_cache_ffn_codegen import generate, extents
    from projected_cache_ffn_reference import (
        reference,
        original_math,
        native_graph,
        packed,
    )
    from host_compiler import executable
    from native_transport import parse_outputs

    source = ROOT / "benchmarks/inference/waferllm/projected_cache_ffn_3x256x512x512_16x16/hls.cpp"
    root = (
        ROOT
        / "validation/evidence"
        / (
            "hls-composed-run-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    print(root.relative_to(ROOT), flush=True)
    shutil.copy2(source, root / "source.cpp")
    shutil.copy2(__file__, root / "driver.py")
    for p in (ROOT / "lib").glob("*.py"):
        d = root / "implementation" / p.name
        d.parent.mkdir(exist_ok=True)
        shutil.copy2(p, d)
    shutil.copytree(ROOT / "include/pragma", root / "implementation/include")
    (root / "implementation/runtime").mkdir()
    shutil.copy2(
        ROOT / "runtime/native/native.cpp",
        root / "implementation/runtime/native.cpp",
    )
    for name, p in [
        ("probe_runtime.py", ROOT / "experiments/probe_runtime.py"),
        ("sdk_process.py", ROOT / "lib/Runtime/sdk_process.py"),
    ]:
        shutil.copy2(p, root / name)
    stage = "frontend"
    try:
        m = parse(source, root)
        s = plan(m)
        generate(s, root)
        (root / "schedule.json").write_text(json.dumps(s, indent=2) + "\n")
        rows = fixtures()
        (root / "batches.json").write_text(json.dumps(rows) + "\n")
        data = (
            str(len(rows))
            + "\n"
            + "".join(
                str(len(row))
                + "\n"
                + "".join(
                    k + " " + str(len(v)) + " " + " ".join(map(str, v)) + "\n"
                    for k, v in row.items()
                )
                for row in rows
            )
        )
        (root / "native-input.txt").write_text(data)
        stage = "native"
        cmd = [
            executable(),
            "-std=c++17",
            "-ffp-contract=off",
            "-DMW_BOUND=2",
            "-DMW_EPOCHS=8",
            "-DMW_MAX_INPUT=131072",
            "-Werror",
            "-Wno-unknown-pragmas",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            "-I",
            str(root / "implementation/include"),
            str(root / "source.cpp"),
            str(root / "implementation/runtime/native.cpp"),
            "-o",
            str(root / "native"),
        ]
        (root / "native-command.json").write_text(json.dumps(cmd) + "\n")
        c = subprocess.run(cmd, capture_output=True, text=True)
        (root / "native-compile.log").write_text(c.stdout + c.stderr)
        assert c.returncode == 0
        c = subprocess.run(
            [str(root / "native")], input=data, capture_output=True, text=True
        )
        (root / "native-output.txt").write_text(c.stdout)
        (root / "native-stderr.txt").write_text(c.stderr)
        assert c.returncode == 0
        native = parse_outputs(c.stdout)
        # Insert read-only native observations at the preserved Clang declarations.
        ast = read(root / "00_clang_ast.json")
        body = next(v for v in ast["inner"] if v["kind"] == "CompoundStmt")
        insert = []
        for node_id, label in roles(s).items():
            decl = next(
                v
                for v in body["inner"]
                if v["kind"] == "DeclStmt"
                and len(v["inner"]) == 1
                and v["inner"][0].get("name") == node_id
            )
            end = decl["range"]["end"]
            offset = end["offset"] + end["tokLen"]
            insert.append(
                (offset, f'\n spatial::output("observe_{label}",{node_id});\n'.encode())
            )
        code = (root / "source.cpp").read_bytes()
        for offset, text in sorted(insert, reverse=True):
            assert code[offset - 1 : offset] == b";"
            code = code[:offset] + text + code[offset:]
        (root / "native-observed.cpp").write_bytes(code)
        ocmd = [
            (
                str(root / "native-observed.cpp")
                if v == str(root / "source.cpp")
                else str(root / "native-observed") if v == str(root / "native") else v
            )
            for v in cmd
        ]
        (root / "observer-command.json").write_text(json.dumps(ocmd) + "\n")
        c = subprocess.run(ocmd, capture_output=True, text=True)
        (root / "observer-compile.log").write_text(c.stdout + c.stderr)
        assert c.returncode == 0
        c = subprocess.run(
            [str(root / "native-observed")], input=data, capture_output=True, text=True
        )
        (root / "observer-output.txt").write_text(c.stdout)
        (root / "observer-stderr.txt").write_text(c.stderr)
        assert c.returncode == 0
        observed = parse_outputs(c.stdout)
        reports = []
        physical = []
        expected = []
        for e, row in enumerate(rows):
            assert {
                k: v for k, v in observed[e].items() if not k.startswith("observe_")
            } == native[e]
            predicted = native_graph(m, row)
            ideal = original_math(s, row)
            checks = {}
            for node_id, label in roles(s).items():
                actual = np.asarray(observed[e]["observe_" + label]).reshape(
                    predicted[node_id].shape
                )
                np.testing.assert_array_equal(
                    actual.astype(np.float16).view(np.uint16),
                    predicted[node_id].astype(np.float16).view(np.uint16),
                    err_msg=f"{e} native IR {label}",
                )
                checks[label] = metric(actual, ideal[label])
            native_mass = float(
                np.max(
                    np.abs(
                        np.asarray(observed[e]["observe_probability"])
                        .reshape(3, 512)
                        .sum(axis=1)
                        - 1
                    )
                )
            )
            reports.append(
                dict(epoch=e, checks=checks, probability_mass_error=native_mass)
            )
            (root / "native-math.json").write_text(json.dumps(reports, indent=2) + "\n")
            assert all(
                v["relative_l2"] <= 0.02 and v["relative_peak"] <= 0.03
                for v in checks.values()
            ), f"fixed native gate epoch {e}"
            assert native_mass <= 0.01, "fixed native probability mass gate"
            stage = "target_reference"
            raw, target = reference(s, row)
            checks = {k: metric(v, ideal[k]) for k, v in target.items()}
            (root / f"target-math-{e}.json").write_text(
                json.dumps(checks, indent=2) + "\n"
            )
            assert all(
                v["relative_l2"] <= 0.02 and v["relative_peak"] <= 0.03
                for v in checks.values()
            ), f"fixed target preflight epoch {e}"
            target_mass = float(np.max(np.abs(target["probability"].sum(axis=1) - 1)))
            (root / f"target-mass-{e}.json").write_text(
                json.dumps(dict(error=target_mass, limit=0.01)) + "\n"
            )
            assert target_mass <= 0.01, "fixed target probability mass gate"
            ports = packed(s, row)
            physical.append({k: v.tolist() for k, v in ports.items()})
            want = {
                k: np.asarray(v, np.float16).view(np.uint16).tolist()
                for k, v in raw.items()
            }
            p = s["attention"]["P"]
            want.update(
                progress=np.full((p, p, 1), e + 1).tolist(),
                stages=np.ones((p, p, 8), int).tolist(),
                attention_progress=np.tile([1] * 10 + [e + 1], (p, p, 1)).tolist(),
            )
            expected.append(want)
            print("NATIVE/IR/MATH/PREFLIGHT", e + 1, flush=True)
        stage = "ready_for_sdk"
        schema = dict(
            rows=16,
            cols=16,
            inputs={k: len(v[0][0]) for k, v in physical[0].items()},
            outputs=extents(s),
            immutable=list(physical[0]),
            progress="progress",
            initialize="init_task",
            launch="hls_main",
        )
        for n, v in [
            ("inputs.json", physical),
            ("expected.json", expected),
            ("schema.json", schema),
            (
                "sdk-command.json",
                [
                    "cslc",
                    "layout.csl",
                    "--arch=wse3",
                    "--fabric-dims=23,18",
                    "--fabric-offsets=4,1",
                    "-o=out",
                    "--memcpy",
                    "--channels=1",
                ],
            ),
            (
                "runtime-options.json",
                dict(suppress_trace=True, num_threads=8, dump_core=False),
            ),
        ]:
            (root / n).write_text(json.dumps(v) + "\n")
    except Exception:
        (root / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        (root / "stage.json").write_text(json.dumps(dict(stage=stage)) + "\n")
        (root / "provenance.json").write_text(
            json.dumps(
                dict(
                    scope="Complete HLS/parent/generated CSL experimental path, 18 native observed stages and fixed mathematical gates; no catalog admission or source-relative performance qualification.",
                    files={
                        str(p.relative_to(root)): sha(p)
                        for p in root.rglob("*")
                        if p.is_file()
                        and "__pycache__" not in p.parts
                        and p.name != "provenance.json"
                    },
                ),
                indent=2,
            )
            + "\n"
        )


def run(root):
    root = root.resolve()
    execute(root, 3600)
    actual = read(root / "results.json")
    expected = read(root / "expected.json")
    assert actual["success"] and len(actual["cases"]) == len(expected) == 8
    for e, (case, want) in enumerate(zip(actual["cases"], expected)):
        for k, v in want.items():
            np.testing.assert_array_equal(case[k], v, err_msg=f"{e} actual raw {k}")
        assert np.all((np.asarray(case["queues"]) & 60) == 60)
    (root / "full-raw-review.json").write_text(
        json.dumps(
            dict(
                passed=True,
                epochs=8,
                results_sha256=sha(root / "results.json"),
                provenance_sha256=sha(root / "provenance.json"),
                scope="Actual full generated graph raw trajectory; independent actual mathematical review and source performance comparison remain required.",
            ),
            indent=2,
        )
        + "\n"
    )
    print("FULL GENERATED CSL RAW PASS", root, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--execute", type=Path)
    p.add_argument("--worker", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.worker:
        mesh_half_worker(a.worker.resolve())
    elif a.execute:
        run(a.execute)
