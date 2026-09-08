"""Probe source probability-times-V layout expectations before resident composition."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, difflib, json, re, shutil, sys
from pathlib import Path
from probe_runtime import execute, mesh_half_worker, read, sha
from attention_layout_adapter import align_column_major_value

ROOT = repository_root(__file__)


def prepare(device_layout=False, m=64, n=128, counters=False):
    assert (m, n) in ((64, 128), (128, 256))
    sys.path.insert(0, str(ROOT / "lib"))
    import numpy as np
    from mesh_common import pack_tiles
    from mesh_twohop import block_index

    root = (
        ROOT
        / "validation/evidence"
        / (
            "attention-value-source-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    src = ROOT / "third_party/sources/waferllm/Prefill/src"
    original = (src / "prefill.csl").read_text()
    text = original
    first = text.index("fn prefill_struct() void {")
    last = text.index(
        "\n// --------------------------------------------------------------------------",
        first,
    )
    text = text[:first] + "fn prefill_struct() void {hls_finish();}\n" + text[last:]
    text = re.sub(r"@export_symbol\([^;]*?\);", "", text)
    first = text.index("fn matmul_compute()")
    last = text.index("fn rmsnorm_x()", first)
    part = text[first:last]
    needle = "        step += 1;"
    assert part.count(needle) == 1
    part = part.replace(
        needle,
        "        @fmovh(@increment_dsd_offset(hd,step*seq_len_p_pe*dim_p_pe,f16),output_dsd);\n"
        + needle,
    )
    text = text[:first] + part + text[last:]
    text += """
var history=@zeros([P*seq_len_p_pe*dim_p_pe]f16);const hd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*dim_p_pe}->history[i]});
var start=@zeros([3]u16);var end=@zeros([3]u16);var timing=@zeros([6]u16);var progress=@zeros([1]u16);
fn hls_main() void {timestamp.enable_tsc();timestamp.get_timestamp(&start);output_matmul();}
fn hls_finish() void {timestamp.get_timestamp(&end);timestamp.disable_tsc();for(@range(i16,3)) |i| {timing[i]=start[i];timing[i+3]=end[i];}progress[0]+=1;sys_mod.unblock_cmd_stream();}
var hp:[*]f16=&history;var tp:[*]u16=&timing;var pp:[*]u16=&progress;
comptime {@export_symbol(ptr_score,"probability");@export_symbol(ptr_XV,"value");@export_symbol(ptr_output,"output");@export_symbol(hp,"history");@export_symbol(tp,"timing");@export_symbol(pp,"progress");@export_symbol(init_task);@export_symbol(hls_main);}
"""
    assert "var ptr_output:" in text
    if device_layout:
        text = align_column_major_value(text)
    if counters:
        copy_line = "        @fmovh(@increment_dsd_offset(hd,step*seq_len_p_pe*dim_p_pe,f16),output_dsd);\n"
        assert text.count(copy_line) == 1
        text = text.replace(copy_line, "").replace(
            "var history=@zeros([P*seq_len_p_pe*dim_p_pe]f16);",
            "var history=@zeros([1]f16);",
        )
    (root / "prefill.csl").write_text(text)
    (root / "source-adapter.diff").write_text(
        "".join(difflib.unified_diff(original.splitlines(True), text.splitlines(True)))
    )
    layout = (src / "layout.csl").read_text()
    layout = re.sub(r"@export_name\([^;]*?\);", "", layout)
    pos = layout.rfind("}")
    layout = (
        layout[:pos]
        + """@export_name("probability",[*]f16,true);@export_name("value",[*]f16,true);@export_name("output",[*]f16,true);@export_name("history",[*]f16,true);@export_name("timing",[*]u16,true);@export_name("progress",[*]u16,true);@export_name("init_task",fn()void);@export_name("hls_main",fn()void);
"""
        + layout[pos:]
    )
    (root / "layout.csl").write_text(layout)
    shutil.copytree(src / "comm_lib", root / "comm_lib")
    base = ROOT / "validation/evidence/score-matmul-source-20260907T015139514873Z"
    for name in ["sdk-command.json", "runtime-options.json", "WaferLLM-LICENSE.txt"]:
        shutil.copyfile(base / name, root / name)
    cmd = read(root / "sdk-command.json")
    mt, nt = m // 8, n // 8
    cmd = [
        (
            f"--params=P:8,dim_p_pe:{nt},pes_p_head:8,pes_p_kv_head:8,head_dim_p_pe:{nt},seq_len_p_pe:{mt},ffn_dim_p_pe:{nt}"
            if x.startswith("--params=")
            else x
        )
        for x in cmd
    ]
    (root / "sdk-command.json").write_text(json.dumps(cmd) + "\n")
    for name, source in [
        ("driver.py", Path(__file__)),
        (
            "attention_layout_adapter.py",
            ROOT / "experiments/attention_layout_adapter.py",
        ),
        ("probe_runtime.py", ROOT / "experiments/probe_runtime.py"),
        ("sdk_process.py", ROOT / "lib/Runtime/sdk_process.py"),
    ]:
        shutil.copyfile(source, root / name)
    p = 8
    mt, nt = m // p, n // p
    rng = np.random.default_rng(210109)
    probability = np.asarray(rng.uniform(0, 1, (m, m)), np.float16).astype(float)
    probability = np.asarray(
        probability / probability.sum(axis=1)[:, None], np.float16
    ).astype(float)
    value = np.asarray(rng.uniform(-0.25, 0.25, (m, n)), np.float16).astype(float)
    pairs = [(np.eye(m), value), (probability, value)]
    batches = []
    cases = []
    for style in (
        ["logical_column_major"]
        if device_layout
        else ["logical_column_major", "column_prealigned_row_major"]
    ):
        for pr, v in pairs:
            vp = pack_tiles(v, p, p, "F")
            if style != "logical_column_major":
                vp = np.stack(
                    [
                        np.stack(
                            [
                                v[
                                    block_index(p, y, x)
                                    * mt : (block_index(p, y, x) + 1)
                                    * mt,
                                    x * nt : (x + 1) * nt,
                                ].ravel(order="C")
                                for x in range(p)
                            ]
                        )
                        for y in range(p)
                    ]
                )
            batches.append(
                dict(probability=pack_tiles(pr, p, p, "F").tolist(), value=vp.tolist())
            )
            cases.append(
                dict(
                    layout=style,
                    probability=pr.ravel().tolist(),
                    value=v.ravel().tolist(),
                )
            )
    for name, data in [
        ("inputs.json", batches),
        ("logical-inputs.json", cases),
        ("geometry.json", dict(M=m, N=n, P=p)),
        (
            "schema.json",
            dict(
                rows=p,
                cols=p,
                inputs=dict(probability=mt * mt, value=mt * nt),
                outputs=dict(
                    output=mt * nt,
                    history=1 if counters else p * mt * nt,
                    timing=6,
                    progress=1,
                ),
                launch="hls_main",
                initialize="init_task",
                progress="progress",
            ),
        ),
    ]:
        (root / name).write_text(json.dumps(data) + "\n")
    files = {str(v.relative_to(root)): sha(v) for v in root.rglob("*") if v.is_file()}
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                device_layout_adapter=device_layout,
                instrumentation="counters" if counters else "sampled",
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                source_sha256=sha(src / "prefill.csl"),
                files=files,
                scope="Original output_matmul source body and communication; isolate host entry/finish and record each prefix. Two input-layout hypotheses, logical prior-stage column-major versus explicitly host-prealigned row-major V. The latter is a diagnostic input adapter, not resident HLS conversion or a fix to full inference.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--execute", type=Path)
    g.add_argument("--worker", type=Path)
    p.add_argument("--device-layout", action="store_true")
    p.add_argument("--large", action="store_true")
    p.add_argument("--counters", action="store_true")
    a = p.parse_args()
    if a.prepare:
        prepare(
            a.device_layout, 128 if a.large else 64, 256 if a.large else 128, a.counters
        )
    elif a.execute:
        execute(a.execute.resolve(), 1800)
    else:
        mesh_half_worker(a.worker.resolve())
