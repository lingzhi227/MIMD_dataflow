"""Actual SDK vector f16→f32 merge→f16 and warm-reset/word-transport probe."""

import argparse, datetime, json, shutil, sys
from pathlib import Path
import numpy as np
from probe_runtime import execute, mesh_half_worker, sha, read, verify

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--prepare", action="store_true")
p.add_argument("--execute", type=Path)
p.add_argument("--worker", type=Path)
p.add_argument("--analyze", type=Path)
a = p.parse_args()
if a.prepare:
    d = (
        ROOT
        / "evidence"
        / (
            "block-accumulate-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    d.mkdir()
    shutil.copyfile(
        ROOT / "toolchain/runtime/block_accumulate.csl", d / "block_accumulate.csl"
    )
    (d / "layout.csl").write_text(
        """const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});
layout {@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});@export_name("input",[*]f16,true);@export_name("history",[*]f16,true);@export_name("wide",[*]u32,true);@export_name("progress",[*]u16,true);@export_name("main",fn()void);}
"""
    )
    (d / "pe.csl").write_text("""param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);const block=@import_module("block_accumulate.csl",.{.length=16});
var input=@zeros([128]f16);var history=@zeros([128]f16);var work=@zeros([16]f16);var progress=@zeros([1]u16);
const iv=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{16}->input[i]});const hv=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{16}->history[i]});const wv=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{16}->work[i]});
fn main() void {block.reset();for(@range(i16,8)) |r| {@fmovh(wv,@increment_dsd_offset(iv,r*16,f16));block.merge(&work);@fmovh(@increment_dsd_offset(hv,r*16,f16),wv);}block.snapshot();progress[0]+=1;sys.unblock_cmd_stream();}
var ip:[*]f16=&input;var hp:[*]f16=&history;var wp:[*]u32=block.words;var pp:[*]u16=&progress;
comptime {@export_symbol(ip,"input");@export_symbol(hp,"history");@export_symbol(wp,"wide");@export_symbol(pp,"progress");@export_symbol(main);}
""")
    rng = np.random.default_rng(210111)
    cases = [
        np.full((8, 16), 28.0),
        rng.uniform(-32, 32, (8, 16)),
        np.zeros((8, 16)),
        np.tile(
            np.array([32, -32, 2**-20, -(2**-20), 1, -1, 0.5, -0.5])[:, None], (1, 16)
        ),
    ]
    cases[3][:, 1] = [32, 2**-10, -32, 2**-10, 1, -1, 2**-12, -(2**-12)]
    data = [
        dict(input=np.asarray(v, np.float16).astype(float).reshape(1, 1, 128).tolist())
        for v in cases
    ]
    schema = dict(
        rows=1,
        cols=1,
        inputs=dict(input=128),
        outputs=dict(input=128, history=128, wide=16, progress=1),
        output_word_bits=dict(wide=32),
        immutable=["input"],
        progress="progress",
        launch="main",
    )
    for name, v in [
        ("schema.json", schema),
        ("inputs.json", data),
        (
            "runtime-options.json",
            dict(suppress_trace=True, num_threads=4, dump_core=True),
        ),
        (
            "sdk-command.json",
            [
                "cslc",
                "layout.csl",
                "--arch=wse3",
                "--fabric-dims=8,3",
                "--fabric-offsets=4,1",
                "-o=out",
                "--memcpy",
                "--channels=1",
            ],
        ),
    ]:
        (d / name).write_text(json.dumps(v) + "\n")
    shutil.copyfile(__file__, d / "driver.py")
    shutil.copyfile(ROOT / "experiments/probe_runtime.py", d / "probe_runtime.py")
    shutil.copyfile(ROOT / "toolchain/sdk_process.py", d / "sdk_process.py")
    (d / "provenance.json").write_text(
        json.dumps(
            dict(
                files={
                    str(f.relative_to(d)): sha(f) for f in d.rglob("*") if f.is_file()
                },
                scope="SDK DSD conversion/merge primitive probe, not complete HLS application qualification",
            ),
            indent=2,
        )
        + "\n"
    )
    print(d.relative_to(ROOT))
elif a.execute:
    execute(a.execute.resolve())
elif a.worker:
    mesh_half_worker(a.worker.resolve())
elif a.analyze:
    d = a.analyze
    verify(d)
    e = read(d / "execution.json")
    assert e["success"] and e["results_sha256"] == sha(d / "results.json")
    r = read(d / "results.json")
    assert r["success"] and r["runtime_instances"] == 1 and len(r["cases"]) == 4
    for b, c in zip(read(d / "inputs.json"), r["cases"]):
        acc = np.zeros(16, np.float32)
        hist = []
        for partial in np.asarray(b["input"], np.float16).reshape(8, 16):
            acc = np.asarray(acc + partial.astype(np.float32), np.float32)
            hist.extend(acc.astype(np.float16).view(np.uint16).tolist())
        np.testing.assert_array_equal(c["history"], np.asarray(hist).reshape(1, 1, 128))
        np.testing.assert_array_equal(c["wide"], acc.view(np.uint32).reshape(1, 1, 16))
    out = d / "numerical-review.json"
    assert not out.exists()
    out.write_text(
        json.dumps(
            dict(
                passed=True,
                calls=4,
                half_prefix_words=512,
                float32_final_values=64,
                scope="exact vector conversion/merge/narrowing, explicit u32 readout of f32 storage, changed warm inputs and reset; no application performance claim",
                results_sha256=sha(d / "results.json"),
                analyzer_sha256=sha(Path(__file__)),
            ),
            indent=2,
        )
        + "\n"
    )
    print(out)
