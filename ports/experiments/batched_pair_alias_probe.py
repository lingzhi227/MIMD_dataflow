"""Actual in-place Q/K slots, nonzero offsets and unchanged adjacent V/canaries."""

import argparse, datetime, json, shutil, sys
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, execute, half_vector_worker

ROOT = Path(__file__).resolve().parents[1]


def prepare(bundle):
    root = (
        ROOT
        / "evidence"
        / (
            "batched-pair-alias-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    shutil.copyfile(
        bundle / "batched_pair_rotation_local.csl",
        root / "batched_pair_rotation_local.csl",
    )
    (root / "pe.csl").write_text("""param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const rotate=@import_module("batched_pair_rotation_local.csl",.{.batches=5,.features=128,.sampled=0,.swapped=1});
const L:i16=640;const W:i16=1924;
var X=@zeros([W]f16);var A=@zeros([W]f16);var B=@zeros([W]f16);var scratch=@zeros([256]f16);var history=@zeros([1]f16);var cosine=@zeros([64]f16);var sine=@zeros([64]f16);var progress=@zeros([1]u16);
const xd=@get_dsd(mem1d_dsd,.{.base_address=&X,.extent=W});const ad=@set_dsd_base_addr(xd,&A);const bd=@set_dsd_base_addr(xd,&B);
fn main() void {
 @fmovh(ad,xd);@fmovh(bd,xd);for(@range(i16,64)) |j| {cosine[j]=0.75;sine[j]=0.5;}
 rotate.apply(@ptrcast([*]f16,&A[1]),@ptrcast([*]f16,&A[1]),&cosine,&sine,&scratch,&history);
 rotate.apply(@ptrcast([*]f16,&A[1+L]),@ptrcast([*]f16,&A[1+L]),&cosine,&sine,&scratch,&history);
 rotate.apply(@ptrcast([*]f16,&X[1]),@ptrcast([*]f16,&B[1]),&cosine,&sine,&scratch,&history);
 rotate.apply(@ptrcast([*]f16,&X[1+L]),@ptrcast([*]f16,&B[1+L]),&cosine,&sine,&scratch,&history);
 progress[0]+=1;sys.unblock_cmd_stream();}
var xp:[*]f16=&X;var ap:[*]f16=&A;var bp:[*]f16=&B;var pp:[*]u16=&progress;
comptime {@export_symbol(xp,"X");@export_symbol(ap,"A");@export_symbol(bp,"B");@export_symbol(pp,"progress");@export_symbol(main);}
""")
    (root / "layout.csl").write_text(
        """const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});
layout {@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});@export_name("X",[*]f16,true);@export_name("A",[*]f16,true);@export_name("B",[*]f16,true);@export_name("progress",[*]u16,true);@export_name("main",fn()void);}
"""
    )
    rng = np.random.default_rng(210109)
    inputs = []
    expected = []
    for epoch in range(8):
        x = np.asarray(rng.uniform(-8, 8, 1924), np.float16)
        x[[0, 1921, 1922, 1923]] = [-13, 17, -19, 23]
        if epoch == 1:
            x[1:1281] = 0
        if epoch == 2:
            x[1:1281] = 8
        y = x.copy()
        for offset in (1, 641):
            a = x[offset : offset + 640].reshape(5, 128)
            b = y[offset : offset + 640].reshape(5, 128)
            q = lambda v: np.asarray(v, np.float16)
            b[:, ::2] = q(q(a[:, 1::2] * 0.75) - q(a[:, ::2] * 0.5))
            b[:, 1::2] = q(q(a[:, ::2] * 0.75) + q(a[:, 1::2] * 0.5))
        inputs.append(x.astype(float).tolist())
        expected.append(y.view(np.uint16).tolist())
    for name, value in {
        "inputs.json": inputs,
        "expected.json": expected,
        "schema.json": dict(input="X", outputs=["A", "B"], length=1924),
        "sdk-command.json": [
            "cslc",
            "layout.csl",
            "--arch=wse3",
            "--fabric-dims=8,3",
            "--fabric-offsets=4,1",
            "-o=out",
            "--memcpy",
            "--channels=1",
        ],
    }.items():
        (root / name).write_text(json.dumps(value) + "\n")
    for path, name in [
        (Path(__file__), "driver.py"),
        (ROOT / "experiments/probe_runtime.py", "probe_runtime.py"),
        (ROOT / "toolchain/sdk_process.py", "sdk_process.py"),
    ]:
        shutil.copyfile(path, root / name)
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                hls_manifest_sha256=sha(bundle / "manifest.json"),
                scope="Primitive: in-place Q/K nonzero slots and out-of-place agreement with unchanged V/canaries; not a new HLS application",
                files={p.name: sha(p) for p in root.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT))


def run(root):
    execute(root, 300)
    r = read(root / "results.json")
    e = read(root / "expected.json")
    assert r["success"] and len(r["cases"]) == len(e) == 8
    for c, w in zip(r["cases"], e):
        np.testing.assert_array_equal(c["A"], w)
        np.testing.assert_array_equal(c["B"], w)
    (root / "alias-review.json").write_text(
        json.dumps(
            dict(
                passed=True,
                epochs=8,
                output_words=8 * 2 * 1924,
                results_sha256=sha(root / "results.json"),
                provenance_sha256=sha(root / "provenance.json"),
                scope="Q and K in-place at nonzero offsets, repeated descriptors and separate output agree; V and edge sentinels unchanged. No cache mutation or head semantics.",
            ),
            indent=2,
        )
        + "\n"
    )
    print("BATCHED PAIR ALIAS PASS", root)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", type=Path)
    p.add_argument("--execute", type=Path)
    p.add_argument("--worker", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.prepare.resolve())
    elif a.execute:
        run(a.execute.resolve())
    elif a.worker:
        half_vector_worker(a.worker.resolve())
