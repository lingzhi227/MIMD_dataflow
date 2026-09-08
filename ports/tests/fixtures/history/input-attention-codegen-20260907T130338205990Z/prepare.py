"""Development HLS lowering probe; native math gates precede real SDK execution.

Uses shared frontend, typed verifier, resource planner and codegen directly while
unified runtime dispatch is under development. Not a catalog qualification.
"""

import argparse, datetime, hashlib, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "toolchain")]


def prepare(diagnose_numerical_failure=False):
    import numpy as np
    from input_attention_source import source
    from input_attention_fixtures import batches, check as math_check
    from frontend import parse
    from mesh_input_attention import verify, plan, generate, inputs
    from mesh_mlp_sdk import parameters, WIDE_PORTS
    from mesh_attention_tail_sdk import extents as tail_extents
    from native_transport import parse_outputs
    from host_compiler import executable

    root = (
        ROOT
        / "evidence"
        / (
            "input-attention-codegen-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    print(root.relative_to(ROOT), flush=True)
    (root / "source.cpp").write_text(source())
    raw = parse(root / "source.cpp", root)
    raw["instrumentation"] = "counters"
    m = verify(raw, 8, 2)
    s = plan(m)
    for name, value in (
        ("frontend.json", raw),
        ("semantic.json", m),
        ("schedule.json", s),
    ):
        (root / name).write_text(json.dumps(value, indent=2) + "\n")
    generate(s, root)
    bs = batches(64, 64, 256, 8)
    for b in bs:
        inputs(m, b)
    (root / "logical-inputs.json").write_text(json.dumps(bs) + "\n")
    data = str(len(bs)) + "\n"
    for b in bs:
        data += str(len(b)) + "\n"
        for name, values in b.items():
            data += (
                name + " " + str(len(values)) + " " + " ".join(map(str, values)) + "\n"
            )
    (root / "native-input.txt").write_text(data)
    shutil.copytree(ROOT / "toolchain/include", root / "include")
    shutil.copyfile(ROOT / "toolchain/runtime/native.cpp", root / "native.cpp")
    cmd = [
        executable(),
        "-std=c++17",
        "-ffp-contract=off",
        "-DMW_BOUND=2",
        "-DMW_EPOCHS=8",
        "-DMW_MAX_INPUT=16384",
        "-Werror",
        "-Wno-unknown-pragmas",
        "-fsanitize=undefined",
        "-fno-sanitize-recover=all",
        "-I",
        str(root / "include"),
        str(root / "source.cpp"),
        str(root / "native.cpp"),
        "-o",
        str(root / "native"),
    ]

    def native(command, stem):
        (root / (stem + "-command.json")).write_text(json.dumps(command) + "\n")
        r = subprocess.run(command, capture_output=True, text=True)
        (root / (stem + "-compile.log")).write_text(r.stdout + r.stderr)
        assert r.returncode == 0, stem
        r = subprocess.run([command[-1]], input=data, capture_output=True, text=True)
        (root / (stem + "-output.txt")).write_text(r.stdout)
        (root / (stem + "-stderr.txt")).write_text(r.stderr)
        assert r.returncode == 0, stem
        return parse_outputs(r.stdout)

    original = native(cmd, "native")
    ast = json.loads((root / "00_clang_ast.json").read_text())
    body = next(v for v in ast["inner"] if v["kind"] == "CompoundStmt")
    indices = dict(
        input_normalized=11,
        q_raw=12,
        k_raw=13,
        v_raw=14,
        q=15,
        k=16,
        score=18,
        probability=19,
        attention=20,
        projection=21,
        delta=28,
    )
    edits = []
    for name, index in indices.items():
        node = m["nodes"][index]
        decl = next(
            v
            for v in body["inner"]
            if v["kind"] == "DeclStmt"
            and len(v["inner"]) == 1
            and v["inner"][0].get("name") == node["id"]
        )
        end = decl["range"]["end"]
        offset = end["offset"] + end["tokLen"]
        edits.append(
            (
                offset,
                (
                    '\n spatial::output("__observe_' + name + '",' + node["id"] + ");\n"
                ).encode(),
            )
        )
    text = (root / "source.cpp").read_bytes()
    for offset, insertion in sorted(edits, reverse=True):
        assert text[offset - 1 : offset] == b";"
        text = text[:offset] + insertion + text[offset:]
    (root / "observed.cpp").write_bytes(text)
    observed_cmd = cmd[:]
    observed_cmd[observed_cmd.index(str(root / "source.cpp"))] = str(
        root / "observed.cpp"
    )
    observed_cmd[-1] = str(root / "observed")
    observed = native(observed_cmd, "observed")
    checks = []
    for b, old, row in zip(bs, original, observed):
        assert {k: v for k, v in row.items() if not k.startswith("__observe_")} == old
        obs = {name: row["__observe_" + name] for name in indices}
        obs["v"] = obs["v_raw"]
        try:
            result = math_check(64, 64, 256, s["epsilon"], s["scale"], b, old, obs)
            checks.append(dict(passed=True, numerical=result))
        except AssertionError as error:
            checks.append(dict(passed=False, error=repr(error)))
    (root / "native-gate.json").write_text(
        json.dumps(
            dict(
                passed=all(c["passed"] for c in checks),
                checks=checks,
                scope="Actual unchanged C++ final outputs plus eleven actual native branch observations; original-eleven-input mathematical oracle.",
            ),
            indent=2,
        )
        + "\n"
    )
    if not all(c["passed"] for c in checks):
        assert (
            diagnose_numerical_failure
        ), "Native numerical gate failed; preserve evidence. Only explicit diagnostic SDK execution may follow, never qualification."
    source_control = ROOT / "evidence/input-attention-source-20260907T122338297913Z"
    physical = json.loads((source_control / "inputs.json").read_text())
    assert bs[:3] == json.loads((source_control / "logical-inputs.json").read_text())
    for row in physical:
        row["residual"] = row.pop("input_x")
    (root / "inputs.json").write_text(json.dumps(physical) + "\n")
    ext = tail_extents(s)
    ext.update(
        q_weight=64,
        k_weight=64,
        v_weight=64,
        cosine=4,
        sine=4,
        input_normalized=64,
        input_q_raw=64,
        input_k_raw=64,
        input_projection_history=1,
        input_left_first=1,
        input_right_first=1,
        input_pair_history=1,
        input_prefix_progress=12,
    )
    lengths = {k: np.asarray(v).shape[-1] for k, v in physical[0].items()}
    schema = dict(
        rows=8,
        cols=8,
        inputs=lengths,
        outputs=ext,
        immutable=list(lengths),
        launch="hls_main",
        initialize="init_task",
        output_word_bits={k: 32 for k in ext if k in WIDE_PORTS},
    )
    (root / "schema.json").write_text(json.dumps(schema) + "\n")
    (root / "sdk-command.json").write_text(
        json.dumps(
            [
                "cslc",
                "layout.csl",
                "--arch=wse3",
                "--fabric-dims=15,10",
                "--fabric-offsets=4,1",
                parameters(s),
                "-o=out",
                "--memcpy",
                "--channels=1",
            ]
        )
        + "\n"
    )
    (root / "runtime-options.json").write_text(
        json.dumps(dict(suppress_trace=True, num_threads=8, dump_core=True)) + "\n"
    )
    for name in ("probe_runtime.py",):
        shutil.copyfile(ROOT / "experiments" / name, root / name)
    shutil.copyfile(ROOT / "toolchain/sdk_process.py", root / "sdk_process.py")
    (root / "driver.py").write_text(
        'import sys\nfrom pathlib import Path\nfrom probe_runtime import execute,mesh_half_worker\nr=Path(sys.argv[2]).resolve()\nif sys.argv[1]=="--worker":mesh_half_worker(r)\nelse:execute(r,2400)\n'
    )
    shutil.copyfile(__file__, root / "prepare.py")
    for name in (
        "input_attention_fixtures.py",
        "attention_tail_fixtures.py",
        "prefill_tail_fixtures.py",
        "feed_forward_fixtures.py",
    ):
        shutil.copyfile(ROOT / name, root / name)
    shutil.copytree(
        ROOT / "toolchain",
        root / "implementation",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                scope="Development 31-node HLS shared codegen; not registered/qualified. Three source-matched SDK diagnostic calls. Eight actual native checks include a preserved cancellation failure; this probe cannot qualify the numerical contract.",
                files=files,
            ),
            indent=2,
        )
        + "\n"
    )
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnose-numerical-failure", action="store_true")
    args = ap.parse_args()
    prepare(args.diagnose_numerical_failure)
