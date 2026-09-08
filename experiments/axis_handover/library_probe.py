"""Exercise reusable SDK axis reducer: odd lengths, map conversion, aliasing."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, math, shutil, struct, sys
from pathlib import Path
import numpy as np

ROOT = repository_root(__file__)
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
        / "validation/evidence"
        / (
            "sdk-axis-library-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    shutil.copyfile(
        ROOT / "runtime/csl/sdk_axis_reduce.csl", root / "sdk_axis_reduce.csl"
    )
    layout = (
        (source / "layout.csl")
        .read_text()
        .replace(
            '@export_name("C",[*]f16,true);',
            '@export_name("C",[*]f16,true); @export_name("D",[*]f16,true);',
        )
    )
    (root / "layout.csl").write_text(layout)
    pe = (source / "pe.csl").read_text()
    a = pe.index("const mx =")
    b = pe.index("var A =", a)
    pe = (
        pe[:a]
        + """const collective=@import_module("sdk_axis_reduce.csl",.{.c2d_params=c2d_params,.capacity=12,.f_callback=after_reduce});
"""
        + pe[b:]
    )
    pe = (
        pe.replace("var A = @zeros([2]f16);", "var A = @zeros([4]f16);")
        .replace(
            "var B = @zeros([8]f16);",
            "var B = @zeros([12]f16);\nvar D = @zeros([12]f16);",
        )
        .replace("var C = @zeros([2]f16);", "var C = @zeros([6]f16);")
    )
    pe = pe.replace("var send = @zeros([8]f32);\nvar reduced = @zeros([8]f32);\n", "")
    a = pe.index("fn widen(")
    b = pe.index("fn init_task()", a)
    pe = pe[:a] + pe[b:]
    pe = pe.replace("mx.init(); my.init();", "collective.init();")
    a = pe.index("    widen(&A,2);")
    b = pe.index("var ap:", a)
    pe = pe[:a] + """    collective.start(1,&A,&A,3);
}
fn after_reduce() void {
    callbacks[0]+=1;
    if (phase==0) {
        skew(1); skew(2); phase=1;
        D[11]=B[11];
        collective.start(0,&B,&D,11);
    } else if (phase==1) {
        skew(3); phase=2;
        collective.start(1,&C,&C,5);
    } else {
        @assert(phase==2 and !collective.busy); phase=3;
        time.get_timestamp(&finish); time.disable_tsc();
        for (@range(u16,3)) |i| { timing[i]=start[i]; timing[i+3]=finish[i]; }
        const qi=config.input_queue_status.get(); const qo=config.output_queue_status.get();
        queues[0]=@as(u16,qi.empty); queues[1]=@as(u16,qo.empty);
        progress[0]+=1; sys.unblock_cmd_stream();
    }
}
""" + pe[b:]
    pe = (
        pe.replace("    @bind_local_task(advance,CALLBACK);\n", "")
        .replace("var ap:[*]f16=&A;", "var outp:[*]f16=&D;\nvar ap:[*]f16=&A;")
        .replace(
            '    @export_symbol(ap,"A");',
            '    @export_symbol(outp,"D");\n    @export_symbol(ap,"A");',
        )
    )
    assert "mx." not in pe and "my." not in pe and "widen(" not in pe
    (root / "pe.csl").write_text(pe)
    inputs = []
    expected = []
    yy, xx = np.indices((8, 8))
    for epoch, (batch, old_expected) in enumerate(
        zip(read(source / "inputs.json"), read(source / "expected.json"))
    ):
        a, b, c = [np.asarray(batch[k]) for k in ["A", "B", "C"]]
        guard = ((1 + epoch) * 0.25 + (xx + yy) * 0.03125)[:, :, None]
        values = dict(
            A=np.concatenate([a, a[:, :, :1] * 0.25, guard], axis=2),
            B=np.concatenate([b, b[:, :, :3] * 0.25, -guard], axis=2),
            C=np.concatenate([c, c * 0.5, c[:, :, :1] * 0.25, guard * 2], axis=2),
        )
        inputs.append({k: v.tolist() for k, v in values.items()})
        row = {k: old_expected[k] for k in ["delays", "delay_value"]}
        row["B"] = values["B"].astype(np.float16).view(np.uint16).tolist()
        for name, length in [("A", 3), ("B", 11), ("C", 5)]:
            v = values[name]
            out = v.astype(np.float16).view(np.uint16).copy()
            for other in range(8):
                for lane in range(length):
                    vector = v[other, :, lane] if name == "B" else v[:, other, lane]
                    ratios = [float(x).as_integer_ratio() for x in vector]
                    den = max(d for _, d in ratios)
                    assert sum(abs(n * (den // d)) for n, d in ratios) <= 2**24
                    word = struct.unpack(
                        "<H", struct.pack("<e", math.fsum(map(float, vector)))
                    )[0]
                    if name == "B":
                        out[other, :, lane] = word
                    else:
                        out[:, other, lane] = word
            row["D" if name == "B" else name] = out.tolist()
        expected.append(row)
    schema = read(source / "schema.json")
    schema["inputs"] = dict(A=4, B=12, C=6)
    schema["outputs"].update(A=4, B=12, C=6, D=12)
    schema["immutable"] = ["B"]
    files = {
        "inputs.json": inputs,
        "expected.json": expected,
        "schema.json": schema,
        "runtime-options.json": read(source / "runtime-options.json"),
        "sdk-command.json": read(source / "sdk-command.json"),
    }
    for name, value in files.items():
        (root / name).write_text(json.dumps(value) + "\n")
    for p, name in [
        (Path(__file__), "driver.py"),
        (ROOT / "experiments/probe_runtime.py", "probe_runtime.py"),
        (ROOT / "lib/Runtime/sdk_process.py", "sdk_process.py"),
    ]:
        shutil.copyfile(p, root / name)
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                scope="Reusable SDK reducer primitive only; odd3/11/5 logical lengths within even4/12/6 transport arrays, changing nonzero canaries, in-place A/C and immutable out-of-place B→D",
                source_probe=str(source.relative_to(ROOT)),
                source_inputs_sha256=sha(source / "inputs.json"),
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
    for epoch, (row, want) in enumerate(zip(r["cases"], e)):
        for name, value in want.items():
            np.testing.assert_array_equal(row[name], value, err_msg=f"{epoch} {name}")
        np.testing.assert_array_equal(row["callbacks"], 3 * (epoch + 1))
        assert np.all((np.asarray(row["queues"]) & 60) == 60)
    (root / "library-review.json").write_text(
        json.dumps(
            dict(
                passed=True,
                epochs=8,
                lengths=[3, 11, 5],
                in_place_and_out_of_place=True,
                nonzero_padding_preserved=True,
                results_sha256=sha(root / "results.json"),
                provenance_sha256=sha(root / "provenance.json"),
                scope="SDK primitive execution with fixed dyadic f32-exact reduction fixtures; not arbitrary floating-point accuracy or HLS application qualification",
            ),
            indent=2,
        )
        + "\n"
    )
    print("SDK AXIS LIBRARY PASS", root)


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
