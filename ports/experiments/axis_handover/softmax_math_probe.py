"""Direct SDK exp/reciprocal contracts needed by positive normalization."""

import argparse, datetime, json, shutil, sys
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
    from sdk_math_reference import exp_f16_nonpositive

    root = (
        ROOT
        / "evidence"
        / (
            "sdk-softmax-math-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    source = ROOT / "evidence/sdk-stable-silu-20260907T194744387145Z"
    for name in ["runtime-options.json", "sdk-command.json"]:
        shutil.copyfile(source / name, root / name)
    (root / "pe.csl").write_text("""param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);const math=@import_module("<math>");
var X=@zeros([128]f16);var D=@zeros([128]f16);var E=@zeros([128]f16);var R=@zeros([128]f16);var progress=@zeros([1]u16);
fn reciprocal(x:f16) f16 {return 1.0/x;}
fn hls_main() void {
 const xd=@get_dsd(mem1d_dsd,.{.base_address=&X,.extent=128});const dd=@get_dsd(mem1d_dsd,.{.base_address=&D,.extent=128});
 const ed=@get_dsd(mem1d_dsd,.{.base_address=&E,.extent=128});const rd=@get_dsd(mem1d_dsd,.{.base_address=&R,.extent=128});
 @map(math.exp_f16,xd,ed);@map(reciprocal,dd,rd);progress[0]+=1;sys.unblock_cmd_stream();
}
var xp:[*]f16=&X;var dp:[*]f16=&D;var ep:[*]f16=&E;var rp:[*]f16=&R;var pp:[*]u16=&progress;
comptime {@export_symbol(xp,"X");@export_symbol(dp,"D");@export_symbol(ep,"E");@export_symbol(rp,"R");@export_symbol(pp,"progress");@export_symbol(hls_main);}
""")
    layout = (
        (source / "layout.csl")
        .read_text()
        .replace(
            '@export_name("Y",[*]f16,true);',
            "".join('@export_name("' + n + '",[*]f16,true);' for n in ["D", "E", "R"]),
        )
    )
    (root / "layout.csl").write_text(layout)
    half = np.arange(65536, dtype=np.uint16).view(np.float16)
    neg = half[np.isfinite(half) & (half <= 0)]
    den = half[(half >= 1) & (half <= 512)]
    assert len(neg) == 31745 and len(den) == 9217
    count = 4 * 8 * 8 * 128
    x = np.pad(neg, (0, count - len(neg))).reshape(4, 8, 8, 128)
    d = np.resize(den, count).reshape(4, 8, 8, 128)
    inputs = [
        dict(X=a.astype(float).tolist(), D=b.astype(float).tolist())
        for a, b in zip(x, d)
    ]
    expected = [
        dict(
            E=np.asarray([exp_f16_nonpositive(v) for v in a.ravel()], np.float16)
            .reshape(8, 8, 128)
            .view(np.uint16)
            .tolist(),
            R=(1 / b.astype(float)).astype(np.float16).view(np.uint16).tolist(),
        )
        for a, b in zip(x, d)
    ]
    for name, v in [
        ("inputs.json", inputs),
        ("expected.json", expected),
        (
            "schema.json",
            dict(
                rows=8,
                cols=8,
                inputs=dict(X=128, D=128),
                outputs=dict(X=128, D=128, E=128, R=128, progress=1),
                immutable=["X", "D"],
                progress="progress",
                launch="hls_main",
            ),
        ),
    ]:
        (root / name).write_text(json.dumps(v) + "\n")
    for path, name in [
        (Path(__file__), "driver.py"),
        (ROOT / "experiments/probe_runtime.py", "probe_runtime.py"),
        (ROOT / "toolchain/sdk_process.py", "sdk_process.py"),
        (ROOT / "toolchain/sdk_math_reference.py", "math-reference.py"),
    ]:
        shutil.copyfile(path, root / name)
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                scope="31745 finite nonpositive half encodings including signed zeros; all9217 half denominators [1,512]; direct exp and reciprocal, primitive only",
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
    expect = read(root / "expected.json")
    assert r["success"] and len(r["cases"]) == len(expect) == 4
    for case, want in zip(r["cases"], expect):
        for name, v in want.items():
            np.testing.assert_array_equal(case[name], v)
        e = np.asarray(case["E"], np.uint16).view(np.float16)
        assert np.all(np.isfinite(e) & (e >= 0) & (e <= 1))
        x = np.asarray(case["X"], np.uint16).view(np.float16)
        assert np.all(e[x == 0] == 1)
    (root / "math-review.json").write_text(
        json.dumps(
            dict(
                passed=True,
                epochs=4,
                exp_finite_nonpositive_encodings=31745,
                exp_bounds=[0, 1],
                exp_zero=1,
                reciprocal_denominator_encodings=9217,
                reciprocal_domain=[1, 512],
                exact_reference=True,
                results_sha256=sha(root / "results.json"),
                provenance_sha256=sha(root / "provenance.json"),
            ),
            indent=2,
        )
        + "\n"
    )
    print("SDK SOFTMAX MATH PASS", root)


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
        mesh_half_worker(a.worker.resolve())
