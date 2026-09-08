"""Execute the explicit-precision public C++ frontend and actual branch outputs.

This proves native authoring/AST semantics, not admitted mixed backend lowering.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import datetime, hashlib, json, shutil, subprocess, sys
from pathlib import Path

ROOT = repository_root(__file__)
sys.path[:0] = [str(ROOT), str(ROOT / "lib")]
from input_attention_mixed_source import source
from frontend import parse
from host_compiler import executable
from native_transport import parse_outputs
from input_attention_fixtures import batches, check


def build():
    root = (
        ROOT
        / "validation/evidence"
        / (
            "input-attention-mixed-frontend-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    print(root.relative_to(ROOT), flush=True)
    (root / "hls.cpp").write_text(source())
    raw = parse(root / "hls.cpp", root)
    (root / "frontend.json").write_text(json.dumps(raw, indent=2) + "\n")
    bs = batches()
    (root / "batches.json").write_text(json.dumps(bs) + "\n")
    data = "8\n"
    for b in bs:
        data += str(len(b)) + "\n"
        for name, v in b.items():
            data += name + " " + str(len(v)) + " " + " ".join(map(str, v)) + "\n"
    (root / "native-input.txt").write_text(data)
    shutil.copytree(ROOT / "include/pragma", root / "include")
    shutil.copyfile(ROOT / "runtime/native/native.cpp", root / "native.cpp")
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
        str(root / "hls.cpp"),
        str(root / "native.cpp"),
        "-o",
        str(root / "native"),
    ]

    def run(command, stem):
        (root / (stem + "-command.json")).write_text(json.dumps(command) + "\n")
        r = subprocess.run(command, capture_output=True, text=True)
        (root / (stem + "-compile.log")).write_text(r.stdout + r.stderr)
        assert r.returncode == 0
        r = subprocess.run([command[-1]], input=data, capture_output=True, text=True)
        (root / (stem + "-output.txt")).write_text(r.stdout)
        (root / (stem + "-stderr.txt")).write_text(r.stderr)
        assert r.returncode == 0
        rows = parse_outputs(r.stdout)
        assert len(rows) == 8
        return rows

    original = run(cmd, "native")
    ast = json.loads((root / "00_clang_ast.json").read_text())
    body = next(n for n in ast["inner"] if n["kind"] == "CompoundStmt")
    variables = dict(
        input_normalized="input_normalized",
        q_raw="q_raw",
        k_raw="k_raw",
        v_raw="v",
        q="q",
        k="k",
        score="score",
        probability="probability",
        attention="attention",
        projection="projection",
        delta="delta",
        z="z",
        normalized_z="x",
    )
    text = (root / "hls.cpp").read_bytes()
    edits = []
    for name, variable in variables.items():
        decl = next(
            v
            for v in body["inner"]
            if v["kind"] == "DeclStmt"
            and len(v["inner"]) == 1
            and v["inner"][0].get("name") == variable
        )
        end = decl["range"]["end"]
        offset = end["offset"] + end["tokLen"]
        assert text[offset - 1 : offset] == b";"
        edits.append(
            (
                offset,
                (
                    '\n spatial::output("__observed_' + name + '",' + variable + ");\n"
                ).encode(),
            )
        )
    for offset, addition in sorted(edits, reverse=True):
        text = text[:offset] + addition + text[offset:]
    (root / "observed.cpp").write_bytes(text)
    cmd[cmd.index(str(root / "hls.cpp"))] = str(root / "observed.cpp")
    cmd[-1] = str(root / "observed")
    observed = run(cmd, "observed")
    checks = []
    for b, row, old in zip(bs, observed, original):
        assert {k: v for k, v in row.items() if not k.startswith("__observed_")} == old
        obs = {k: row["__observed_" + k] for k in variables}
        obs["v"] = obs["v_raw"]
        checks.append(check(64, 64, 256, 1e-6, 0.125, b, old, obs))
    (root / "driver.py").write_bytes(Path(__file__).read_bytes())
    report = dict(
        passed=True,
        scope=__doc__,
        checks=checks,
        files={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file()
        },
    )
    (root / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    return root


if __name__ == "__main__":
    build()
