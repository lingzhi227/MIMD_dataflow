"""Compare independently allocated SDK collective planes with the ring probe.

The f32 reduction and final narrowing intentionally differ from grouped half
association. Use original-input mathematics; do not claim bitwise equivalence.
"""

import argparse
import datetime
import json
import math
import shutil
import struct
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = (
    [str(HERE)] if (HERE / "probe_runtime.py").exists() else [str(ROOT / "experiments")]
)
from probe_runtime import execute, mesh_half_worker, read, sha, verify


def prepare(source):
    source = source.resolve()
    verify(source)
    assert read(source / "execution.json")["success"]
    root = (
        ROOT
        / "evidence"
        / (
            "sdk-independent-planes-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    shutil.copyfile(HERE / "sdk_planes.csl", root / "pe.csl")
    (root / "layout.csl").write_text(
        """const memcpy=@import_module("<memcpy/get_params>",.{.width=8,.height=8});
const c2d=@import_module("<collectives_2d/params>");
layout {
    @set_rectangle(8,8);
    for(@range(u16,8)) |y| {for(@range(u16,8)) |x| {
        @set_tile_code(x,y,"pe.csl",.{.memcpy_params=memcpy.get_params(x),
            .c2d_params=c2d.get_params(x,y, .{
                .x_colors=.{@get_color(0),@get_color(1)},
                .x_entrypoints=.{@get_local_task_id(14),@get_local_task_id(15)},
                .y_colors=.{@get_color(4),@get_color(5)},
                .y_entrypoints=.{@get_local_task_id(16),@get_local_task_id(17)}
            })
        });
    }}
    @export_name("A",[*]f16,true); @export_name("B",[*]f16,true);
    @export_name("C",[*]f16,true); @export_name("progress",[*]u16,true);
    @export_name("queues",[*]u16,true); @export_name("callbacks",[*]u16,true);
    @export_name("delays",[*]u16,true); @export_name("timing",[*]u16,true);
    @export_name("delay_value",[*]f16,true);
    @export_name("init_task",fn()void); @export_name("hls_main",fn()void);
}
"""
    )
    bs = read(source / "inputs.json")
    expected = read(source / "expected.json")
    for b, e in zip(bs, expected):
        for name in ("A", "B", "C"):
            a = np.asarray(b[name])
            result = np.zeros(a.shape, np.uint16)
            for other in range(8):
                for lane in range(a.shape[2]):
                    v = a[other, :, lane] if name == "B" else a[:, other, lane]
                    # Directed dyadic fixtures fit exactly in binary32 throughout
                    # either SDK linear tree. Final narrowing is the only rounding.
                    ratios = [float(item).as_integer_ratio() for item in v]
                    denominator = max(pair[1] for pair in ratios)
                    # Binary16 denominators are powers of two. Every partial
                    # subset sum is exact in f32 when this integer bound fits.
                    units = [num * (denominator // den) for num, den in ratios]
                    assert sum(map(abs, units)) <= 2**24
                    value = math.fsum(map(float, v))
                    assert float(np.float32(value)) == value
                    word = struct.unpack("<H", struct.pack("<e", value))[0]
                    if name == "B":
                        result[other, :, lane] = word
                    else:
                        result[:, other, lane] = word
            e[name] = result.tolist()
    schema = read(source / "schema.json")
    schema["outputs"]["callbacks"] = schema["outputs"].pop("sequence")
    command = [
        v for v in read(source / "sdk-command.json") if not v.startswith("--params=")
    ]
    for name, value in {
        "inputs.json": bs,
        "expected.json": expected,
        "schema.json": schema,
        "runtime-options.json": read(source / "runtime-options.json"),
        "sdk-command.json": command,
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
                scope="SDK X/Y independent planes, callback-owned f32 buffers with f16 I/O. Different reduction precision from ring baseline; no HLS/application qualification.",
                source_probe=str(source.relative_to(ROOT)),
                source_inputs_sha256=sha(source / "inputs.json"),
                source_results_sha256=sha(source / "results.json"),
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
    r, e = read(root / "results.json"), read(root / "expected.json")
    assert r["success"] and len(r["cases"]) == len(e) == 8
    cycles = []
    for epoch, (row, expected) in enumerate(zip(r["cases"], e)):
        for name, want in expected.items():
            np.testing.assert_array_equal(row[name], want, err_msg=f"{epoch} {name}")
        np.testing.assert_array_equal(row["callbacks"], 6 * (epoch + 1))
        assert np.all((np.asarray(row["queues"]) & 60) == 60)
        t = np.asarray(row["timing"], np.int64)
        d = sum((t[:, :, i + 3] - t[:, :, i]) * (1 << (16 * i)) for i in range(3)) % (
            1 << 48
        )
        assert np.all((d > 0) & (d < 2**32))
        cycles.append(d.tolist())
    (root / "plane-review.json").write_text(
        json.dumps(
            dict(
                passed=True,
                epochs=8,
                cycles_per_pe=cycles,
                resources=dict(
                    colors=[0, 1, 4, 5],
                    input_queues=[2, 3, 4, 5],
                    output_queues=[2, 3, 4, 5],
                    local_tasks=[10, 14, 15, 16, 17],
                ),
                scope="Bounded f32 SDK collective alternative with same input/skew counts; distinct arithmetic, schedule and task resources, not an isolated barrier cost or application performance claim",
                results_sha256=sha(root / "results.json"),
                provenance_sha256=sha(root / "provenance.json"),
            ),
            indent=2,
        )
        + "\n"
    )
    print("SDK PLANES PASS", root)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", type=Path)
    p.add_argument("--worker", type=Path)
    p.add_argument("--execute", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.prepare)
    elif a.worker:
        mesh_half_worker(a.worker.resolve())
    elif a.execute:
        run(a.execute)
