"""Read-only tensor observations inserted at preserved Clang declaration ranges.

The diagnostic executable uses the frozen header/runtime and the exact same
source bytes plus one output-copy statement. It must preserve original outputs.
This is a native diagnostic, never an extra synthesized host port.
"""

import hashlib, json, subprocess
from pathlib import Path
from frontend import check
from integrity import verify_bundle
from native_transport import parse_outputs


def observe(bundle, node_id, dest):
    bundle = Path(bundle).resolve()
    dest = Path(dest).resolve()
    verify_bundle(bundle)
    check(not dest.exists(), "fresh native observation directory")
    m = json.loads((bundle / "semantic.json").read_text())
    node = next((n for n in m["nodes"] if n["id"] == node_id), None)
    check(
        node is not None and node.get("dtype") in ("f16", "f32", "u32"),
        "observable native tensor",
    )
    ast = json.loads((bundle / "00_clang_ast.json").read_text())
    body = next(v for v in ast["inner"] if v["kind"] == "CompoundStmt")
    declarations = [
        v
        for v in body["inner"]
        if v["kind"] == "DeclStmt"
        and len(v["inner"]) == 1
        and v["inner"][0].get("name") == node_id
    ]
    check(len(declarations) == 1, "one Clang declaration for observed value")
    end = declarations[0]["range"]["end"]
    offset = end["offset"] + end["tokLen"]
    source = (bundle / "source.cpp").read_bytes()
    check(
        source[offset - 1 : offset] == b";", "observation follows complete declaration"
    )
    host = "__native_observed"
    while host in {n.get("host") for n in m["nodes"]}:
        host += "_"
    insertion = ('\n spatial::output("' + host + '",' + node_id + ");\n").encode()
    dest.mkdir(parents=True)
    (dest / "input-manifest.json").write_bytes((bundle / "manifest.json").read_bytes())
    (dest / "driver.py").write_bytes(Path(__file__).read_bytes())
    (dest / "source.cpp").write_bytes(source[:offset] + insertion + source[offset:])
    command = json.loads((bundle / "native-command.json").read_text())
    check(command.count(m["source"]) == 1, "original native source binding")
    command[command.index(m["source"])] = str(dest / "source.cpp")
    command[command.index("-I") + 1] = str(bundle / "implementation/include")
    runtimes = [i for i, v in enumerate(command) if v.endswith(("/runtime/native.cpp", "/runtime/native/native.cpp"))]
    check(len(runtimes) == 1, "frozen native runtime binding")
    command[runtimes[0]] = str(bundle / "implementation/runtime/native.cpp")
    command[command.index("-o") + 1] = str(dest / "native")
    (dest / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, text=True, capture_output=True)
    (dest / "compile.log").write_text(result.stdout + result.stderr)
    check(result.returncode == 0, "native observer compilation failed")
    result = subprocess.run(
        [str(dest / "native")],
        input=(bundle / "native-input.txt").read_text(),
        text=True,
        capture_output=True,
    )
    (dest / "stdout.txt").write_text(result.stdout)
    (dest / "stderr.txt").write_text(result.stderr)
    check(result.returncode == 0, "native observer execution failed")
    rows = parse_outputs(result.stdout)
    original = parse_outputs((bundle / "native-output.txt").read_text())
    check(len(rows) == len(original) == m["epochs"], "native observation epochs")
    for row, old in zip(rows, original):
        check(
            host in row and len(row[host]) == node["shape"][0] * node["shape"][1],
            "native observed shape",
        )
        check(
            {k: v for k, v in row.items() if k != host} == old,
            "native observer changed original outputs",
        )
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(
        passed=True,
        new_sdk_execution=False,
        node=node_id,
        host=host,
        epochs=len(rows),
        source_insertion_byte=offset,
        scope="Actual native C++ tensor output; only a read-only output-copy statement added at the preserved Clang declaration end. Original outputs exactly preserved; frozen headers/runtime and original inputs used. Not a synthesized CSL observer.",
        bundle_manifest_sha256=sha(bundle / "manifest.json"),
        source_sha256=sha(bundle / "source.cpp"),
        driver_sha256=sha(Path(__file__)),
        files={str(p.relative_to(dest)): sha(p) for p in dest.iterdir() if p.is_file()},
    )
    (dest / "observation.json").write_text(json.dumps(report, indent=2) + "\n")
    return [row[host] for row in rows], report
