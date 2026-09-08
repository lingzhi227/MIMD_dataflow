"""Non-associative finite-half inputs distinguish SDK f32 reduction orders."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, shutil, sys
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
HERE = Path(__file__).resolve().parent
sys.path.insert(
    0, str(HERE if (HERE / "probe_runtime.py").exists() else ROOT / "experiments")
)
from probe_runtime import execute, mesh_half_worker, read, sha, verify


def reduce(a, axis, reverse):
    v = np.moveaxis(a, axis, 0)
    if reverse:
        v = v[::-1]
    total = np.asarray(v[0], np.float32)
    for x in v[1:]:
        total = np.asarray(total + x.astype(np.float32), np.float32)
    return total.astype(np.float16)


def prepare():
    source = ROOT / "validation/evidence/sdk-axis-library-20260907T192718252830Z"
    verify(source)
    root = (
        ROOT
        / "validation/evidence"
        / (
            "sdk-axis-association-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    for name in (
        "layout.csl",
        "pe.csl",
        "sdk_axis_reduce.csl",
        "schema.json",
        "runtime-options.json",
        "sdk-command.json",
    ):
        shutil.copyfile(source / name, root / name)
    inputs = []
    expected = []
    forward = []
    pattern = np.array([65504, -65504, 2**-10, 2**-12, 0, 0, 0, 0], np.float16)
    for epoch in range(8):
        batch = {}
        want = {}
        other = {}
        for name, out, length, axis in [
            ("A", "A", 3, 0),
            ("B", "D", 11, 1),
            ("C", "C", 5, 0),
        ]:
            a = np.empty((8, 8, length + 1), np.float16)
            for j in range(length):
                v = np.roll(pattern, (epoch + j) % 8)
                a[:, :, j] = v[:, None] if axis == 0 else v[None, :]
            a[:, :, -1] = np.arange(64).reshape(8, 8) / 64 + (epoch + 1)
            batch[name] = a.astype(float).tolist()

            def broadcast(reverse):
                r = reduce(a[:, :, :length], axis, reverse)
                r = np.repeat(np.expand_dims(r, axis), 8, axis=axis)
                return (
                    np.concatenate([r, a[:, :, -1:]], axis=2).view(np.uint16).tolist()
                )

            want[out] = broadcast(True)
            other[out] = broadcast(False)
            if name == "B":
                want["B"] = a.view(np.uint16).tolist()
        inputs.append(batch)
        expected.append(want)
        forward.append(other)
    files = {"inputs.json": inputs, "expected.json": expected, "forward.json": forward}
    for name, value in files.items():
        (root / name).write_text(json.dumps(value) + "\n")
    for source, name in [
        (Path(__file__), "driver.py"),
        (ROOT / "experiments/probe_runtime.py", "probe_runtime.py"),
        (ROOT / "lib/Runtime/sdk_process.py", "sdk_process.py"),
    ]:
        shutil.copyfile(source, root / name)
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                scope="Directed non-associative SDK f32 root0 chain order probe, not an application",
                files={p.name: sha(p) for p in root.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT))


def run(root):
    execute(root, 600)
    r = read(root / "results.json")
    expect = read(root / "expected.json")
    forward = read(root / "forward.json")
    differences = 0
    assert r["success"] and len(r["cases"]) == 8
    for e, (row, want, other) in enumerate(zip(r["cases"], expect, forward)):
        for name, a in want.items():
            np.testing.assert_array_equal(row[name], a, err_msg=f"{e} {name}")
        for name, a in other.items():
            differences += np.count_nonzero(np.asarray(row[name]) != np.asarray(a))
        np.testing.assert_array_equal(row["callbacks"], 3 * (e + 1))
        assert np.all((np.asarray(row["queues"]) & 60) == 60)
    assert differences > 0
    (root / "association-review.json").write_text(
        json.dumps(
            dict(
                passed=True,
                epochs=8,
                reverse_order_exact=True,
                forward_order_word_differences=int(differences),
                results_sha256=sha(root / "results.json"),
                provenance_sha256=sha(root / "provenance.json"),
                scope="Directed distinguishable root0 reverse-linear f32 reduction cases on X/Y; no universal arbitrary-order equivalence",
            ),
            indent=2,
        )
        + "\n"
    )
    print("SDK AXIS ASSOCIATION PASS", root)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--worker", type=Path)
    p.add_argument("--execute", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.worker:
        mesh_half_worker(a.worker.resolve())
    elif a.execute:
        run(a.execute.resolve())
