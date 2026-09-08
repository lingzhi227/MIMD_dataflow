"""Execute pinned SDK f32 sqrt/exp and range-reduction casts on boundary samples."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, shutil, sys
from pathlib import Path
import numpy as np
from probe_runtime import execute, mesh_half_worker, verify, sha

ROOT = repository_root(__file__)


def prepare(exp_limit=80):
    assert 8 <= exp_limit <= 80
    root = (
        ROOT
        / "validation/evidence"
        / (
            "f32-math-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    names = ["sqrt_value", "exp_value", "cast_value"]
    exports = "".join('@export_name("' + n + '",[*]f32,true);' for n in names)
    (root / "layout.csl").write_text(
        'const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});\nlayout {@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});@export_name("sx",[*]f16,true);@export_name("ex",[*]f16,true);'
        + exports
        + '@export_name("progress",[*]u16,true);@export_name("main",fn()void);}\n'
    )
    pe = """param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);const math=@import_module("<math>");
var sx=@zeros([1024]f16);var ex=@zeros([1024]f16);var progress=@zeros([1]u16);
var sqrt_value=@zeros([256]f32);var exp_value=@zeros([256]f32);var cast_value=@zeros([256]f32);
fn decode(a:[*]f16,i:i16) f32 {var word:u32=0;for(@range(i16,4)) |j| {word|=@as(u32,a[4*i+j])<<@as(u16,8*j);}return @bitcast(f32,word);}
fn main() void {for(@range(i16,256)) |i| {const x=decode(&sx,i);const z=decode(&ex,i);sqrt_value[i]=math.sqrt_f32(x);exp_value[i]=math.exp_f32(z);cast_value[i]=@as(f32,@as(i16,z*@bitcast(f32,@as(u32,0x40b8aa3b))));}progress[0]+=1;sys.unblock_cmd_stream();}
var sp:[*]f16=&sx;var ep:[*]f16=&ex;var pp:[*]u16=&progress;
comptime {@export_symbol(sp,"sx");@export_symbol(ep,"ex");@export_symbol(pp,"progress");@export_symbol(main);}
"""
    for n in names:
        pe += (
            "var "
            + n
            + "_ptr:[*]f32=&"
            + n
            + ";comptime {@export_symbol("
            + n
            + '_ptr,"'
            + n
            + '");}\n'
        )
    (root / "pe.csl").write_text(pe)
    rng = np.random.default_rng(210164)
    sqrt = np.ldexp(
        rng.uniform(1, 2, 4096).astype(np.float32), np.resize(np.arange(-24, 17), 4096)
    ).astype(np.float32)
    sqrt[:8] = np.array(
        [
            2**-24,
            1e-6,
            np.nextafter(np.float32(1e-6), np.float32(0)),
            1,
            2,
            4,
            65504,
            2**16,
        ],
        np.float32,
    )
    exp = np.linspace(-exp_limit, 0, 4096, dtype=np.float32)
    exp[:8] = np.array(
        [0, -0.0, -(2**-24), -exp_limit, -1, -0.125, -0.5, -8], np.float32
    )

    def encode(a):
        return a.astype("<f4").view(np.uint8).astype(float).reshape(1, 1, 1024).tolist()

    batches = [
        dict(sx=encode(sqrt[i : i + 256]), ex=encode(exp[i : i + 256]))
        for i in range(0, 4096, 256)
    ]
    values = [
        ("inputs.json", batches),
        (
            "logical-inputs.json",
            dict(sqrt=sqrt.astype(float).tolist(), exp=exp.astype(float).tolist()),
        ),
        (
            "schema.json",
            dict(
                rows=1,
                cols=1,
                inputs=dict(sx=1024, ex=1024),
                outputs=dict(
                    sx=1024,
                    ex=1024,
                    sqrt_value=256,
                    exp_value=256,
                    cast_value=256,
                    progress=1,
                ),
                output_word_bits={n: 32 for n in names},
                immutable=["sx", "ex"],
                progress="progress",
                launch="main",
            ),
        ),
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
    ]
    for name, v in values:
        (root / name).write_text(json.dumps(v) + "\n")
    shutil.copyfile(__file__, root / "driver.py")
    shutil.copyfile(ROOT / "experiments/probe_runtime.py", root / "probe_runtime.py")
    shutil.copyfile(ROOT / "lib/Runtime/sdk_process.py", root / "sdk_process.py")
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                scope=__doc__,
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
    print(root.relative_to(ROOT), flush=True)


def analyze(root):
    verify(root)
    r = json.loads((root / "results.json").read_text())
    assert r["success"] and len(r["cases"]) == 16
    inp = json.loads((root / "logical-inputs.json").read_text())

    def values(name):
        return np.concatenate(
            [
                np.asarray(row[name], np.uint32).view(np.float32).ravel()
                for row in r["cases"]
            ]
        ).astype(float)

    a, b = values("sqrt_value"), values("exp_value")
    x, z = np.array(inp["sqrt"]), np.array(inp["exp"])
    sr = float(np.max(np.abs(a / np.sqrt(x) - 1)))
    er = float(np.max(np.abs(b / np.exp(z) - 1)))
    assert sr <= 2**-16 and er <= 2e-6 and np.all(b > 0) and np.all(np.isfinite(a))
    c = np.array([0x40B8AA3B], np.uint32).view(np.float32)[0]
    expected = np.trunc(np.float32(z) * c).astype(np.float32)
    np.testing.assert_array_equal(values("cast_value"), expected)
    assert np.all(b[z == 0] == 1)
    out = root / "review.json"
    assert not out.exists()
    out.write_text(
        json.dumps(
            dict(
                passed=True,
                sqrt_samples=len(x),
                exp_samples=len(z),
                sqrt_max_relative_error=sr,
                exp_max_relative_error=er,
                truncating_cast_exact=True,
                scope="Actual SDK boundary/sample observations, not exhaustive binary32 domain proof.",
                results_sha256=sha(root / "results.json"),
            ),
            indent=2,
        )
        + "\n"
    )
    print(out.read_text())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--exp-limit", type=float, default=80)
    p.add_argument("--execute", type=Path)
    p.add_argument("--worker", type=Path)
    p.add_argument("--analyze", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.exp_limit)
    elif a.worker:
        mesh_half_worker(a.worker.resolve())
    elif a.execute:
        execute(a.execute.resolve(), 1200)
    elif a.analyze:
        analyze(a.analyze.resolve())
