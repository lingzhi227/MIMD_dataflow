"""SDK MAX sharing existing SUM planes, both axes and odd extent canaries."""

import argparse, datetime, json, shutil, sys
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
            "sdk-axis-max-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    for name in [
        "layout.csl",
        "sdk_axis_reduce.csl",
        "runtime-options.json",
        "sdk-command.json",
        "schema.json",
    ]:
        shutil.copyfile(source / name, root / name)
    shutil.copyfile(
        ROOT / "toolchain/runtime/sdk_axis_max.csl", root / "sdk_axis_max.csl"
    )
    pe = (source / "pe.csl").read_text()
    pos = pe.index("var A =")
    pe = (
        pe[:pos]
        + """const maximum=@import_module("sdk_axis_max.csl",.{.participants=8,.capacity=11,.f_gather=gather,.f_broadcast=broadcast,.f_callback=after_reduce});
fn gather(axis:u16, input:[*]u32, output:[*]u32, count:u16, callback:local_task_id) void {
 @assert(!collective.busy);
 if(axis==0){collective.mx.gather(0,input,output,count,callback);}
 else{collective.my.gather(0,input,output,count,callback);}
}
fn broadcast(axis:u16, data:[*]u32, count:u16, callback:local_task_id) void {
 if(axis==0){collective.mx.broadcast(0,data,count,callback);}
 else{collective.my.broadcast(0,data,count,callback);}
}
"""
        + pe[pos:]
    )
    pe = pe.replace(
        "collective.start(0,&B,&D,11);", "maximum.start(0,@as(u16,px),&B,&D,11);"
    )
    pe = pe.replace(
        "collective.start(1,&C,&C,5);", "maximum.start(1,@as(u16,py),&C,&C,5);"
    )
    pe = pe.replace(
        "phase==2 and !collective.busy",
        "phase==2 and !collective.busy and !maximum.busy",
    )
    (root / "pe.csl").write_text(pe)
    inputs = read(source / "inputs.json")
    expected = read(source / "expected.json")
    yy, xx = np.indices((8, 8))
    for epoch, (batch, want) in enumerate(zip(inputs, expected)):
        for name, length, axis in [("B", 11, 1), ("C", 5, 0)]:
            a = np.asarray(batch[name], dtype=np.float16)
            for lane in range(length):
                # All-negative values, with the winning coordinate changing by lane/epoch.
                distance = ((xx if axis == 1 else yy) - epoch - lane) % 8
                a[:, :, lane] = (
                    -(1 + distance) * 0.125 - (yy if axis == 1 else xx) * 0.015625
                )
            batch[name] = a.astype(float).tolist()
            out = a.copy()
            out[:, :, :length] = np.max(a[:, :, :length], axis=axis, keepdims=True)
            want["D" if name == "B" else name] = out.view(np.uint16).tolist()
            if name == "B":
                want["B"] = a.view(np.uint16).tolist()
    for name, value in [("inputs.json", inputs), ("expected.json", expected)]:
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
                scope="SUM Y then MAX X then MAX Y on shared SDK modules; odd3/11/5 extents, in-place/out-of-place, all-negative remote maxima, skew and warm repeats; primitive only",
                source_probe=str(source.relative_to(ROOT)),
                source_provenance_sha256=sha(source / "provenance.json"),
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
    execute(root, 900)
    r, e = read(root / "results.json"), read(root / "expected.json")
    assert r["success"] and len(r["cases"]) == len(e) == 8
    for epoch, (row, want) in enumerate(zip(r["cases"], e)):
        for name, value in want.items():
            np.testing.assert_array_equal(row[name], value, err_msg=f"{epoch} {name}")
        np.testing.assert_array_equal(row["callbacks"], 3 * (epoch + 1))
        assert np.all((np.asarray(row["queues"]) & 60) == 60)
    (root / "max-review.json").write_text(
        json.dumps(
            dict(
                passed=True,
                epochs=8,
                shared_planes=True,
                max_axes=["x", "y"],
                all_negative_maxima=True,
                logical_lengths=[3, 11, 5],
                canaries_preserved=True,
                results_sha256=sha(root / "results.json"),
                provenance_sha256=sha(root / "provenance.json"),
                scope="Primitive conformance, not application performance",
            ),
            indent=2,
        )
        + "\n"
    )
    print("SDK AXIS MAX PASS", root)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", type=Path)
    p.add_argument("--execute", type=Path)
    p.add_argument("--worker", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.prepare)
    elif a.execute:
        run(a.execute)
    elif a.worker:
        mesh_half_worker(a.worker.resolve())
