"""Replay original and observed C++ on another host using frozen headers/runtime."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, shutil, subprocess, sys, traceback, hashlib
from pathlib import Path

ROOT = repository_root(__file__)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(bundle):
    sys.path.insert(0, str(bundle / "implementation"))
    from integrity import verify_bundle

    verify_bundle(bundle)
    s = json.loads((bundle / "schedule.json").read_text())
    sys.path.insert(0, str(ROOT / "experiments/composed_ffn"))
    from run import roles

    r = (
        ROOT
        / "validation/evidence"
        / (
            "composed-native-host-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    r.mkdir()
    shutil.copytree(bundle / "implementation", r / "implementation")
    for n in (
        "source.cpp",
        "00_clang_ast.json",
        "native-input.txt",
        "native-output.txt",
        "batches.json",
        "schedule.json",
        "manifest.json",
    ):
        shutil.copy2(
            bundle / n,
            r / ("original-" + n if n in ("native-output.txt", "manifest.json") else n),
        )
    (r / "roles.json").write_text(json.dumps(roles(s)) + "\n")
    shutil.copy2(__file__, r / "driver.py")
    (r / "provenance.json").write_text(
        json.dumps(
            dict(
                scope="Two-host native C++ identity and eighteen observed original-input stages, no additional SDK simulation.",
                bundle_manifest_sha256=sha(bundle / "manifest.json"),
                files={
                    str(p.relative_to(r)): sha(p)
                    for p in r.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print(r.relative_to(ROOT), flush=True)


def run(root):
    root = root.resolve()
    sys.path.insert(0, str(root / "implementation"))
    from native_transport import parse_outputs
    from projected_cache_ffn_reference import original_math, accuracy, native_graph
    import numpy as np

    pr = json.loads((root / "provenance.json").read_text())
    for n, h in pr["files"].items():
        assert sha(root / n) == h, n
    assert not (root / "report.json").exists()
    (root / "compiler-version.txt").write_text(
        subprocess.check_output(["/usr/bin/clang++-17", "--version"], text=True)
    )
    data = (root / "native-input.txt").read_text()
    s = json.loads((root / "schedule.json").read_text())
    roles = json.loads((root / "roles.json").read_text())
    source = (root / "source.cpp").read_bytes()
    ast = json.loads((root / "00_clang_ast.json").read_text())
    body = next(v for v in ast["inner"] if v["kind"] == "CompoundStmt")
    insert = []
    for node_id, label in roles.items():
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
    for offset, value in sorted(insert, reverse=True):
        assert source[offset - 1 : offset] == b";"
        source = source[:offset] + value + source[offset:]
    (root / "observed.cpp").write_bytes(source)
    outputs = []
    for name in ("source", "observed"):
        cmd = [
            "/usr/bin/clang++-17",
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
            str(root / (name + ".cpp")),
            str(root / "implementation/runtime/native.cpp"),
            "-o",
            str(root / (name + ".native")),
        ]
        (root / (name + "-command.json")).write_text(json.dumps(cmd) + "\n")
        c = subprocess.run(cmd, capture_output=True, text=True)
        (root / (name + "-compile.log")).write_text(c.stdout + c.stderr)
        assert c.returncode == 0
        c = subprocess.run(
            [str(root / (name + ".native"))], input=data, capture_output=True, text=True
        )
        (root / (name + "-output.txt")).write_text(c.stdout)
        (root / (name + "-stderr.txt")).write_text(c.stderr)
        assert c.returncode == 0
        outputs.append(parse_outputs(c.stdout))
    original = parse_outputs((root / "original-native-output.txt").read_text())
    assert outputs[0] == original
    batches = json.loads((root / "batches.json").read_text())
    reports = []
    for e, (plain, obs, batch) in enumerate(zip(*outputs, batches)):
        assert {k: v for k, v in obs.items() if not k.startswith("observe_")} == plain
        expected = native_graph(s["graph"]["module"], batch)
        ideal = original_math(s, batch)
        checks = {}
        for node, label in roles.items():
            actual = np.asarray(obs["observe_" + label]).reshape(expected[node].shape)
            np.testing.assert_array_equal(
                actual.astype(np.float16).view(np.uint16),
                expected[node].astype(np.float16).view(np.uint16),
            )
            checks[label] = accuracy(actual, ideal[label])
        mass = float(
            np.max(
                np.abs(
                    np.asarray(obs["observe_probability"]).reshape(3, 512).sum(axis=1)
                    - 1
                )
            )
        )
        assert mass <= 0.01
        reports.append(dict(epoch=e, checks=checks, probability_mass_error=mass))
    assert len(reports) == 8
    (root / "report.json").write_text(
        json.dumps(
            dict(
                passed=True,
                epochs=8,
                native_cross_host_exact=True,
                cases=reports,
                files={
                    str(p.relative_to(root)): sha(p)
                    for p in root.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print("NATIVE HOST PASS", root, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", type=Path)
    p.add_argument("--execute", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.prepare)
    elif a.execute:
        try:
            run(a.execute)
        except BaseException:
            (a.execute / "failure.txt").write_text(traceback.format_exc())
            raise
