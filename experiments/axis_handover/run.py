"""Prepare/execute a bounded, skewed Y/X/Y route handover protocol probe."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = repository_root(__file__)
HERE = Path(__file__).resolve().parent
sys.path[:0] = (
    [str(HERE)]
    if (HERE / "probe_runtime.py").exists()
    else [str(ROOT / "experiments"), str(ROOT / "lib")]
)
from probe_runtime import execute, mesh_half_worker, read, sha


def prepare(groups):
    from grouped_collective_csl import generate
    from reference_handover import axis_words

    assert groups in (2, 4)
    root = (
        ROOT
        / "validation/evidence"
        / (
            "axis-handover-g"
            + str(groups)
            + "-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    generate(root)
    for source, name in [
        (HERE / "reference_handover.py", "reference_handover.py"),
        (HERE / "README.md", "README.md"),
        (ROOT / "runtime/csl/waferllm-LICENSE.txt", "WaferLLM-LICENSE.txt"),
    ]:
        shutil.copyfile(source, root / name)
    shutil.copyfile(
        Path(__file__).with_name("control_ring.csl"), root / "control_ring.csl"
    )
    (root / "pe.csl").write_text("""param memcpy_params;
param groups: i16;
param even: bool;
const P: i16 = 8;
const size: i16 = P / groups;
const sys = @import_module("<memcpy/memcpy>", memcpy_params);
const pos = @import_module("<layout>");
const config = @import_module("<tile_config>");
const time = @import_module("<time>");
const comm = @import_module("axis_grouped_reduce_dynamic.csl", .{
    .P=P, .bsz=8, .pe_num_p_group=size, .root_1st_phase=size/2,
    .root_2nd_phase=(groups/2)*size+size/2,
    .reduce_1st_color_0=@get_color(8), .reduce_1st_color_1=@get_color(7),
    .reduce_2nd_color_0=@get_color(6), .reduce_2nd_color_1=@get_color(5),
    .broadcast_color=@get_color(9)
});
const control = @import_module("control_ring.csl", .{ .P=P, .even=even });
var A = @zeros([2]f16);
var B = @zeros([8]f16);
var C = @zeros([2]f16);
var progress = @zeros([1]u16);
var queues = @zeros([2]u16);
var sequence = @zeros([1]u16);
var delays = @zeros([4]u16);
var timing = @zeros([6]u16);
var start = @zeros([3]u16);
var finish = @zeros([3]u16);
var delay_value = @zeros([1]f16);
const delay_dsd = @get_dsd(mem1d_dsd, .{ .base_address=&delay_value, .extent=1 });
var px: i16 = 0;
var py: i16 = 0;

fn skew(stage: i16) void {
    const epoch = @as(i16, progress[0]);
    var count: i16 = (px*11 + py*7 + epoch*13 + stage*17) % 31;
    if (px == (epoch+stage)%P and py == (epoch*3+stage)%P) { count += 127; }
    delays[stage] = @as(u16, count);
    @fmovh(delay_dsd, 0.0);
    for (@range(i16, count)) |_| { @faddh(delay_dsd, delay_dsd, 1.0); }
}
fn init_task() void {
    px = @as(i16, pos.get_x_coord());
    py = @as(i16, pos.get_y_coord());
    comm.init(px, py, px/size, px%size, py/size, py%size);
    control.init(px, py);
    sys.unblock_cmd_stream();
}
fn hls_main() void {
    time.enable_tsc();
    time.get_timestamp(&start);
    skew(0);
    comm.set_extent(2);
    comm.all_reduce_bsz(py, py/size, py%size, &A);
    control.begin();
    skew(1);
    comm.reconfig_allreduce_axis(0);
    control.end();
    skew(2);
    comm.set_extent(8);
    comm.all_reduce_bsz(px, px/size, px%size, &B);
    control.begin();
    skew(3);
    comm.reconfig_allreduce_axis(1);
    control.end();
    comm.set_extent(2);
    comm.all_reduce_bsz(py, py/size, py%size, &C);
    time.get_timestamp(&finish);
    time.disable_tsc();
    for (@range(i16, 3)) |i| { timing[i]=start[i]; timing[i+3]=finish[i]; }
    const qi = config.input_queue_status.get();
    const qo = config.output_queue_status.get();
    queues[0]=@as(u16, qi.empty); queues[1]=@as(u16, qo.empty);
    sequence[0]=@as(u16, control.sequence);
    progress[0]+=1;
    sys.unblock_cmd_stream();
}
var ap:[*]f16=&A; var bp:[*]f16=&B; var cp:[*]f16=&C;
var pp:[*]u16=&progress; var qp:[*]u16=&queues; var sp:[*]u16=&sequence;
var dp:[*]u16=&delays; var tp:[*]u16=&timing; var vp:[*]f16=&delay_value;
comptime {
    @export_symbol(ap,"A"); @export_symbol(bp,"B"); @export_symbol(cp,"C");
    @export_symbol(pp,"progress"); @export_symbol(qp,"queues");
    @export_symbol(sp,"sequence"); @export_symbol(dp,"delays");
    @export_symbol(tp,"timing"); @export_symbol(vp,"delay_value");
    @export_symbol(init_task); @export_symbol(hls_main);
}
""")
    (root / "layout.csl").write_text("""param groups:i16;
const memcpy=@import_module("<memcpy/get_params>", .{ .width=8, .height=8 });
layout {
    @set_rectangle(8,8);
    for (@range(i16,8)) |y| { for (@range(i16,8)) |x| {
        @set_tile_code(x,y,"pe.csl", .{
            .memcpy_params=memcpy.get_params(x), .groups=groups, .even=(x+y)%2==0
        });
    }}
    @export_name("A",[*]f16,true); @export_name("B",[*]f16,true);
    @export_name("C",[*]f16,true); @export_name("progress",[*]u16,true);
    @export_name("queues",[*]u16,true); @export_name("sequence",[*]u16,true);
    @export_name("delays",[*]u16,true); @export_name("timing",[*]u16,true);
    @export_name("delay_value",[*]f16,true);
    @export_name("init_task",fn()void); @export_name("hls_main",fn()void);
}
""")
    batches, expected = [], []
    yy, xx = np.indices((8, 8))
    for epoch in range(8):
        pattern = np.array([2048, 0, 1, -2048 if epoch % 2 == 0 else 1.0]) * 2 ** (
            -(epoch // 2)
        )
        y = np.zeros(8)
        y[:4] = pattern if groups == 2 else 0
        if groups == 4:
            y[1::2] = pattern
        slab = (
            y[:, None, None]
            * np.array([1, 0.5, -1, -0.5, 2, 0.25, -2, -0.25])[None, :, None]
        )
        a = np.concatenate([slab, -slab], axis=2)
        b = np.concatenate([a, a * 0.5, a * 2, a * 0.25], axis=2).swapaxes(0, 1).copy()
        c = a[::-1].copy()
        values = dict(A=a, B=b, C=c)
        batches.append({k: v.tolist() for k, v in values.items()})
        row = {}
        for key, value in values.items():
            row[key] = axis_words(value, groups, 1 if key == "B" else 0).tolist()
        delays = np.stack(
            [
                (xx * 11 + yy * 7 + epoch * 13 + stage * 17) % 31
                + 127 * ((xx == (epoch + stage) % 8) & (yy == (epoch * 3 + stage) % 8))
                for stage in range(4)
            ],
            axis=2,
        )
        row.update(
            delays=delays.tolist(),
            delay_value=delays[:, :, 3:4].astype(np.float16).view(np.uint16).tolist(),
        )
        expected.append(row)
    files = {
        "inputs.json": batches,
        "expected.json": expected,
        "schema.json": dict(
            rows=8,
            cols=8,
            inputs=dict(A=2, B=8, C=2),
            outputs=dict(
                A=2,
                B=8,
                C=2,
                progress=1,
                queues=2,
                sequence=1,
                delays=4,
                timing=6,
                delay_value=1,
            ),
            progress="progress",
            initialize="init_task",
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
            f"--params=groups:{groups}",
            "-o=out",
            "--memcpy",
            "--channels=1",
        ],
    }
    for name, value in files.items():
        (root / name).write_text(json.dumps(value) + "\n")
    for src, name in [
        (Path(__file__), "driver.py"),
        (ROOT / "experiments/probe_runtime.py", "probe_runtime.py"),
        (ROOT / "lib/Runtime/sdk_process.py", "sdk_process.py"),
    ]:
        shutil.copyfile(src, root / name)
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                groups=groups,
                scope="Experimental static control-ring handover baseline only; no application or production performance qualification",
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
    result, expected = read(root / "results.json"), read(root / "expected.json")
    assert result["success"] and len(result["cases"]) == len(expected) == 8
    cycles = []
    for epoch, (row, want) in enumerate(zip(result["cases"], expected)):
        for key, value in want.items():
            np.testing.assert_array_equal(row[key], value, err_msg=f"{epoch} {key}")
        np.testing.assert_array_equal(np.asarray(row["sequence"]), (epoch + 1) * 6)
        assert np.all((np.asarray(row["queues"]) & 252) == 252)
        t = np.asarray(row["timing"], np.int64)
        duration = sum(
            (t[:, :, i + 3] - t[:, :, i]) * (1 << (16 * i)) for i in range(3)
        ) % (1 << 48)
        assert np.all((duration > 0) & (duration < 2**32))
        cycles.append(duration.tolist())
    (root / "handover-review.json").write_text(
        json.dumps(
            dict(
                passed=True,
                epochs=8,
                axis_sequence=["y", "x", "y"],
                extent_sequence=[2, 8, 2],
                cycles_per_pe=cycles,
                scope="Bounded skewed protocol test including delay and handover overhead; not application throughput or general race-freedom proof",
                results_sha256=sha(root / "results.json"),
                provenance_sha256=sha(root / "provenance.json"),
            ),
            indent=2,
        )
        + "\n"
    )
    print("HANDOVER PROBE PASS", root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=int)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--execute", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    elif args.worker:
        mesh_half_worker(args.worker.resolve())
    elif args.execute:
        run(args.execute)
