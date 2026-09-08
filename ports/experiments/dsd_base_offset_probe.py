"""Execute base-reset semantics for an offset-one strided half DSD."""

import argparse, datetime, json, shutil
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, execute, half_vector_worker

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    root = (
        ROOT
        / "evidence"
        / (
            "dsd-base-offset-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    (root / "pe.csl").write_text(
        """param memcpy_params;const sys=@import_module("<memcpy/memcpy>",memcpy_params);
var X=@zeros([8]f16);var A=@zeros([8]f16);var B=@zeros([8]f16);var progress=@zeros([1]u16);
var odd=@get_dsd(mem1d_dsd,.{.base_address=&X,.offset=1,.extent=4,.stride=2});
const xd=@get_dsd(mem1d_dsd,.{.base_address=&X,.extent=8});const ad=@set_dsd_base_addr(xd,&A);const bd=@set_dsd_base_addr(xd,&B);
const a4=@set_dsd_length(ad,4);const b4=@set_dsd_length(bd,4);
fn main() void {@fmovh(ad,xd);@fmovh(bd,xd);odd=@set_dsd_base_addr(odd,&X);@fmovh(a4,odd);odd=@increment_dsd_offset(odd,1,f16);@fmovh(b4,odd);progress[0]+=1;sys.unblock_cmd_stream();}
var xp:[*]f16=&X;var ap:[*]f16=&A;var bp:[*]f16=&B;var pp:[*]u16=&progress;
comptime {@export_symbol(xp,"X");@export_symbol(ap,"A");@export_symbol(bp,"B");@export_symbol(pp,"progress");@export_symbol(main);}
"""
    )
    (root / "layout.csl").write_text(
        """const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});layout {@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});@export_name("X",[*]f16,true);@export_name("A",[*]f16,true);@export_name("B",[*]f16,true);@export_name("progress",[*]u16,true);@export_name("main",fn()void);}
"""
    )
    inputs = [
        [float((i + 1) * (-1 if i % 2 == 0 else 1) + e) for i in range(8)]
        for e in range(8)
    ]
    for name, v in {
        "inputs.json": inputs,
        "schema.json": dict(input="X", outputs=["A", "B"], length=8),
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
                scope="Actual SDK strided half DSD offset after base reset and explicit restoration, repeated calls",
                files={p.name: sha(p) for p in root.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT))


def run(root):
    execute(root, 180)
    r = read(root / "results.json")
    inputs = read(root / "inputs.json")
    assert r["success"] and len(r["cases"]) == len(inputs) == 8
    for c, x in zip(r["cases"], inputs):
        words = np.asarray(x, np.float16).view(np.uint16).tolist()
        np.testing.assert_array_equal(c["A"], words[::2] + words[4:])
        np.testing.assert_array_equal(c["B"], words[1::2] + words[4:])
    (root / "offset-review.json").write_text(
        json.dumps(
            dict(
                passed=True,
                epochs=8,
                results_sha256=sha(root / "results.json"),
                provenance_sha256=sha(root / "provenance.json"),
                finding="@set_dsd_base_addr resets the effective offset to the supplied address; restore +1 explicitly for odd half elements; stride and extent retained",
            ),
            indent=2,
        )
        + "\n"
    )
    print("DSD BASE OFFSET PASS", root)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--execute", type=Path)
    p.add_argument("--worker", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.execute:
        run(a.execute.resolve())
    elif a.worker:
        half_vector_worker(a.worker.resolve())
