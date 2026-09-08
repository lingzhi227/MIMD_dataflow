"""Actual half underflow signs in direct versus locally float-merged contractions."""

import argparse, datetime, json, shutil, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = (
    [str(HERE)] if (HERE / "probe_runtime.py").exists() else [str(ROOT / "experiments")]
)
from probe_runtime import execute, half_vector_worker, read, sha


def prepare():
    root = (
        ROOT
        / "evidence"
        / (
            "blocked-matmul-signed-zero-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    source = (
        ROOT
        / "projects/waferllm/cache_attention_5x256x512_8x8/run-20260907T220401939966Z"
    )
    for name in ["batched_matmul_local.csl", "batched_matmul_blocked.csl"]:
        shutil.copyfile(source / name, root / name)
    (root / "pe.csl").write_text("""param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const direct=@import_module("batched_matmul_blocked.csl",.{.batches=2,.inner=2,.columns=2,.block_size=2});
const blocked=@import_module("batched_matmul_blocked.csl",.{.batches=2,.inner=2,.columns=2,.block_size=1});
var X=@zeros([4]f16);var A=@zeros([4]f16);var B=@zeros([4]f16);var progress=@zeros([1]u16);
var W=[4]f16{0.0,0.0,0.5,0.5};
fn main() void {direct.compute(&X,@ptrcast([*]f16,&W),&A);blocked.compute(&X,@ptrcast([*]f16,&W),&B);progress[0]+=1;sys.unblock_cmd_stream();}
var xp:[*]f16=&X;var ap:[*]f16=&A;var bp:[*]f16=&B;var pp:[*]u16=&progress;
comptime {@export_symbol(xp,"X");@export_symbol(ap,"A");@export_symbol(bp,"B");@export_symbol(pp,"progress");@export_symbol(main);}
""")
    (root / "layout.csl").write_text(
        """const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});
layout {@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});@export_name("X",[*]f16,true);@export_name("A",[*]f16,true);@export_name("B",[*]f16,true);@export_name("progress",[*]u16,true);@export_name("main",fn()void);}
"""
    )
    eta = 2**-24
    inputs = [[0.0, -k * eta, 0.0, k * eta] for k in [1, 3, 5, 7, 2, 4, 6, 1]]
    # One final half product per row. Float merge starts with +0 and erases -0.
    expected = []
    for x in inputs:
        a = np.repeat(np.asarray([x[1] * 0.5, x[3] * 0.5], np.float16), 2)
        b = (np.zeros(4, np.float32) + a.astype(np.float32)).astype(np.float16)
        expected.append(
            dict(A=a.view(np.uint16).tolist(), B=b.view(np.uint16).tolist())
        )
    files = {
        "inputs.json": inputs,
        "expected.json": expected,
        "schema.json": dict(input="X", outputs=["A", "B"], length=4),
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
    }
    for name, v in files.items():
        (root / name).write_text(json.dumps(v) + "\n")
    for path, name in [
        (Path(__file__), "driver.py"),
        (ROOT / "experiments/probe_runtime.py", "probe_runtime.py"),
        (ROOT / "toolchain/sdk_process.py", "sdk_process.py"),
    ]:
        shutil.copyfile(path, root / name)
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                scope="Direct half final underflow versus explicit float block merge signed zero; exact source runtime library, primitive only",
                hls_manifest_sha256=sha(source / "manifest.json"),
                files={p.name: sha(p) for p in root.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT))


def run(root):
    root = root.resolve()
    execute(root, 180)
    r = read(root / "results.json")
    e = read(root / "expected.json")
    assert r["success"] and len(r["cases"]) == len(e) == 8
    for got, want in zip(r["cases"], e):
        for key, value in want.items():
            np.testing.assert_array_equal(got[key], value)
    (root / "signed-zero-review.json").write_text(
        json.dumps(
            dict(
                passed=True,
                epochs=8,
                direct_negative_zero=r["cases"][0]["A"][:2],
                float_merged_zero=r["cases"][0]["B"][:2],
                results_sha256=sha(root / "results.json"),
                provenance_sha256=sha(root / "provenance.json"),
                scope="Target fast path must preserve half -0; native blocked semantics may produce +0",
            ),
            indent=2,
        )
        + "\n"
    )
    print("SIGNED ZERO PASS", root)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--execute", type=Path)
    p.add_argument("--worker", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.execute:
        run(a.execute)
    elif a.worker:
        half_vector_worker(a.worker.resolve())
