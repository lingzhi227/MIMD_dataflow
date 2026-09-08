"""Exhaustive finite-half SDK conformance for stable-sign SiLU."""

import argparse, datetime, json, math, shutil, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = (
    [str(HERE)]
    if (HERE / "probe_runtime.py").exists()
    else [str(ROOT / "experiments"), str(ROOT / "toolchain")]
)
from probe_runtime import execute, mesh_half_worker, read, sha


def prepare():
    from sdk_math_reference import stable_silu_f16

    root = (
        ROOT
        / "evidence"
        / (
            "sdk-stable-silu-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    shutil.copyfile(
        ROOT / "toolchain/runtime/sdk_stable_silu.csl", root / "sdk_stable_silu.csl"
    )
    shutil.copyfile(
        ROOT / "toolchain/sdk_math_reference.py", root / "math-reference.py"
    )
    (root / "pe.csl").write_text("""param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const silu=@import_module("sdk_stable_silu.csl");
var X=@zeros([128]f16);var Y=@zeros([128]f16);var progress=@zeros([1]u16);
const xd=@get_dsd(mem1d_dsd,.{.base_address=&X,.extent=128});
const yd=@get_dsd(mem1d_dsd,.{.base_address=&Y,.extent=128});
fn hls_main() void {@map(silu.value,xd,yd);progress[0]+=1;sys.unblock_cmd_stream();}
var xp:[*]f16=&X;var yp:[*]f16=&Y;var pp:[*]u16=&progress;
comptime {@export_symbol(xp,"X");@export_symbol(yp,"Y");@export_symbol(pp,"progress");@export_symbol(hls_main);}
""")
    (root / "layout.csl").write_text(
        """const memcpy=@import_module("<memcpy/get_params>",.{.width=8,.height=8});
layout {@set_rectangle(8,8);for(@range(u16,8)) |y| {for(@range(u16,8)) |x| {@set_tile_code(x,y,"pe.csl",.{.memcpy_params=memcpy.get_params(x)});}}
@export_name("X",[*]f16,true);@export_name("Y",[*]f16,true);@export_name("progress",[*]u16,true);@export_name("hls_main",fn()void);}
"""
    )
    all_values = np.arange(65536, dtype=np.uint16).view(np.float16)
    finite = all_values[np.isfinite(all_values)]
    assert len(finite) == 63488
    values = np.concatenate(
        [finite, np.zeros(65536 - len(finite), np.float16)]
    ).reshape(8, 8, 8, 128)
    expected = []
    for batch in values:
        output = np.asarray(
            [stable_silu_f16(v) for v in batch.ravel()], np.float16
        ).reshape(8, 8, 128)
        expected.append(output.view(np.uint16).tolist())
    files = {
        "inputs.json": [dict(X=v.astype(float).tolist()) for v in values],
        "expected.json": expected,
        "schema.json": dict(
            rows=8,
            cols=8,
            inputs=dict(X=128),
            outputs=dict(X=128, Y=128, progress=1),
            immutable=["X"],
            progress="progress",
            launch="hls_main",
        ),
        "runtime-options.json": dict(
            suppress_trace=True, num_threads=4, dump_core=False
        ),
        "sdk-command.json": [
            "cslc",
            "layout.csl",
            "--arch=wse3",
            "--fabric-dims=15,10",
            "--fabric-offsets=4,1",
            "-o=out",
            "--memcpy",
            "--channels=1",
        ],
    }
    for n, v in files.items():
        (root / n).write_text(json.dumps(v) + "\n")
    for p, n in [
        (Path(__file__), "driver.py"),
        (ROOT / "experiments/probe_runtime.py", "probe_runtime.py"),
        (ROOT / "toolchain/sdk_process.py", "sdk_process.py"),
    ]:
        shutil.copyfile(p, root / n)
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                scope="All63488 finite half encodings, plus2048 zero padding; exact SDK model conformance and separately reported standard-math error, not an all-domain relative-accuracy claim",
                files={
                    str(p.relative_to(root)): sha(p)
                    for p in root.rglob("*")
                    if p.is_file()
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT))


def run(root):
    root = root.resolve()
    execute(root, 600)
    r = read(root / "results.json")
    expected = read(root / "expected.json")
    inputs = read(root / "inputs.json")
    assert r["success"] and len(r["cases"]) == len(expected) == 8
    max_absolute = 0.0
    max_ulp = 0
    lost_tails = 0
    for row, want, b in zip(r["cases"], expected, inputs):
        np.testing.assert_array_equal(row["Y"], want)
        x = np.asarray(b["X"])
        y = np.asarray(row["Y"], np.uint16).view(np.float16).astype(float)
        assert np.all(np.isfinite(y)) and np.all(np.abs(y) <= np.abs(x))
        for v, actual in zip(x.ravel(), y.ravel()):
            e = math.exp(-abs(float(v)))
            standard = v / (1 + e) if v >= 0 else v * e / (1 + e)
            max_absolute = max(max_absolute, abs(float(actual) - standard))
            ideal = np.float16(standard)
            iw = int(np.asarray(ideal).view(np.uint16))
            aw = int(np.asarray(np.float16(actual)).view(np.uint16))
            order = lambda w: (~w & 65535) if w & 32768 else w + 32768
            max_ulp = max(max_ulp, abs(order(iw) - order(aw)))
            lost_tails += int(actual == 0 and ideal != 0)
    (root / "silu-review.json").write_text(
        json.dumps(
            dict(
                passed=True,
                epochs=8,
                finite_encodings=63488,
                max_absolute_error=max_absolute,
                max_half_ulp_from_rounded_standard=max_ulp,
                nonzero_standard_half_tails_lost=lost_tails,
                finite_and_magnitude_bounded=True,
                results_sha256=sha(root / "results.json"),
                provenance_sha256=sha(root / "provenance.json"),
                scope="Exhaustive finite-half intrinsic conformance; negative-tail underflow remains explicit, application accuracy must be checked independently",
            ),
            indent=2,
        )
        + "\n"
    )
    print("STABLE SILU PASS", root)


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
        run(a.execute)
