"""Probe resident QK-transpose followed by explicitly repaired stable softmax."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, difflib, json, re, shutil
from pathlib import Path
from probe_runtime import execute, mesh_half_worker, read, sha

ROOT = repository_root(__file__)


def repair_softmax_maximum(body):
    maximum = "    @load_to_dsr(comp_dest_dsr_1, local_max_dsd, .{ .save_address = false});\n    @fmovh(comp_dest_dsr_1, 0.0); // Clearing"
    total = "    @load_to_dsr(comp_dest_dsr_1, local_sum_dsd, .{ .save_address = false});\n    @fmovh(comp_dest_dsr_1, 0.0); // Clearing"
    assert body.count(maximum) == body.count(total) == 1
    result = body.replace(maximum, maximum.replace("0.0", "-65504.0"))
    assert result.count(total) == 1 and result.count("-65504.0") == 1
    return result


def prepare(m=64, n=128, p=8, counters=False):
    assert (m, n, p) in ((64, 128, 8), (128, 256, 8))
    mt, nt = m // p, n // p
    import sys

    sys.path.insert(0, str(ROOT / "lib"))
    import numpy as np
    from mesh_common import pack_tiles

    root = (
        ROOT
        / "validation/evidence"
        / (
            "score-softmax-source-"
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
    text = text[:first] + "fn prefill_struct() void { hls_finish(); }\n" + text[last:]
    # Remove unused public roots so unrelated inference allocations are eliminated.
    text = re.sub(r"@export_symbol\([^;]*?\);", "", text)
    needle = "        step += 1;\n        matmul_T_reduce_add();"
    assert text.count(needle) == 1
    text = text.replace(
        needle,
        """        @fmovh(@increment_dsd_offset(hls_hd,step*seq_len_p_pe*seq_len_p_pe,f16),seqLen_seqLen_tmp_dsd);
        step += 1;
        matmul_T_reduce_add();""",
    )
    text += """
var hls_history=@zeros([P*seq_len_p_pe*seq_len_p_pe]f16);
const hls_hd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*seq_len_p_pe}->hls_history[i]});
var hls_start=@zeros([3]u16);var hls_end=@zeros([3]u16);var hls_time=@zeros([6]u16);var hls_progress=@zeros([1]u16);
fn hls_main() void {timestamp.enable_tsc();timestamp.get_timestamp(&hls_start);score_matmul();}
fn hls_finish() void {timestamp.get_timestamp(&hls_end);timestamp.disable_tsc();for(@range(i16,3)) |i| {hls_time[i]=hls_start[i];hls_time[i+3]=hls_end[i];}hls_progress[0]+=1;sys_mod.unblock_cmd_stream();}
var hls_hp:[*]f16=&hls_history;var hls_tp:[*]u16=&hls_time;var hls_pp:[*]u16=&hls_progress;
comptime {@export_symbol(ptr_XQ,"q");@export_symbol(ptr_XK,"k");@export_symbol(ptr_score,"score");@export_symbol(hls_hp,"history");@export_symbol(hls_tp,"timing");@export_symbol(hls_pp,"progress");@export_symbol(init_task);@export_symbol(hls_main);}
"""
    first = text.index("fn softmax_score()")
    last = text.index("fn output_matmul()", first)
    body = text[first:last]
    body = repair_softmax_maximum(body)
    text = text[:first] + body + text[last:]
    text = text.replace(
        "fn hls_main() void {timestamp.enable_tsc();",
        "fn hls_main() void {hls_phase=0;hls_scale[0]=alpha;timestamp.enable_tsc();",
    )
    text = text.replace(
        "fn hls_finish() void {timestamp.get_timestamp",
        "fn hls_finish() void {if(hls_phase==0){@fmovh(hls_logits_dsd,score_dsd);hls_phase=1;softmax_score();return;}timestamp.get_timestamp",
    )
    text += """
var hls_phase:i16=0;var hls_logits=@zeros([seq_len_p_pe*seq_len_p_pe]f16);var hls_scale=@zeros([1]f16);
const hls_logits_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*seq_len_p_pe}->hls_logits[i]});
var hls_lp:[*]f16=&hls_logits;var hls_ap:[*]f16=&hls_scale;
comptime {@export_symbol(hls_lp,"logits");@export_symbol(hls_ap,"scale");@export_symbol(ptr_local_max,"peak");@export_symbol(ptr_local_sum,"inverse");}
"""
    if counters:
        text = text.replace("@fmovh(hls_logits_dsd,score_dsd);", "")
        text = text.replace(
            "var hls_logits=@zeros([seq_len_p_pe*seq_len_p_pe]f16);",
            "var hls_logits=@zeros([1]f16);",
        )
        text = text.replace(
            "        @fmovh(@increment_dsd_offset(hls_hd,step*seq_len_p_pe*seq_len_p_pe,f16),seqLen_seqLen_tmp_dsd);\n",
            "",
        )
        text = text.replace(
            "var hls_history=@zeros([P*seq_len_p_pe*seq_len_p_pe]f16);",
            "var hls_history=@zeros([1]f16);",
        )
    (root / "prefill.csl").write_text(text)
    (root / "source-adapter.diff").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                text.splitlines(True),
                fromfile="pinned/prefill.csl",
                tofile="score-probe/prefill.csl",
            )
        )
    )
    layout = (src / "layout.csl").read_text()
    layout = re.sub(r"@export_name\([^;]*?\);", "", layout)
    pos = layout.rfind("}")
    layout = (
        layout[:pos]
        + """@export_name("logits",[*]f16,true);@export_name("scale",[*]f16,true);@export_name("peak",[*]f16,true);@export_name("inverse",[*]f16,true);@export_name("q",[*]f16,true);@export_name("k",[*]f16,true);@export_name("score",[*]f16,true);@export_name("history",[*]f16,true);@export_name("timing",[*]u16,true);@export_name("progress",[*]u16,true);@export_name("init_task",fn()void);@export_name("hls_main",fn()void);
"""
        + layout[pos:]
    )
    (root / "layout.csl").write_text(layout)
    shutil.copytree(src / "comm_lib", root / "comm_lib")
    base = ROOT / "validation/evidence/prefill-rms-qkv-20260907T011935115666Z"
    for name in ["sdk-command.json", "runtime-options.json", "WaferLLM-LICENSE.txt"]:
        shutil.copyfile(base / name, root / name)
    command = read(root / "sdk-command.json")
    command = [
        (
            f"--params=P:{p},dim_p_pe:{nt},pes_p_head:{p},pes_p_kv_head:{p},head_dim_p_pe:{nt},seq_len_p_pe:{mt},ffn_dim_p_pe:{nt}"
            if v.startswith("--params=")
            else v
        )
        for v in command
    ]
    (root / "sdk-command.json").write_text(json.dumps(command) + "\n")
    for name, source in [
        ("driver.py", Path(__file__)),
        ("probe_runtime.py", ROOT / "experiments/probe_runtime.py"),
        ("sdk_process.py", ROOT / "lib/Runtime/sdk_process.py"),
    ]:
        shutil.copyfile(source, root / name)
    rng = np.random.default_rng(210108)
    pairs = [
        (
            np.eye(m, n, dtype=np.float16),
            np.roll(np.eye(m, n, dtype=np.float16), 3, axis=0),
        ),
        (
            rng.uniform(-0.25, 0.25, (m, n)).astype(np.float16),
            rng.uniform(-0.25, 0.25, (m, n)).astype(np.float16),
        ),
        (
            np.zeros((m, n), np.float16),
            rng.uniform(-0.25, 0.25, (m, n)).astype(np.float16),
        ),
    ]
    inputs = [
        dict(
            q=pack_tiles(q, p, p, "F").astype(float).tolist(),
            k=pack_tiles(k, p, p, "F").astype(float).tolist(),
        )
        for q, k in pairs
    ]
    for name, value in [
        ("inputs.json", inputs),
        (
            "schema.json",
            dict(
                rows=p,
                cols=p,
                inputs=dict(q=mt * nt, k=mt * nt),
                outputs=dict(
                    q=mt * nt,
                    k=mt * nt,
                    score=mt * mt,
                    logits=1 if counters else mt * mt,
                    scale=1,
                    peak=mt,
                    inverse=mt,
                    history=1 if counters else p * mt * mt,
                    timing=6,
                    progress=1,
                ),
                immutable=["q"],
                launch="hls_main",
                initialize="init_task",
                progress="progress",
            ),
        ),
        ("geometry.json", dict(M=m, N=n, P=p)),
    ]:
        (root / name).write_text(json.dumps(value) + "\n")
    files = {str(v.relative_to(root)): sha(v) for v in root.rglob("*") if v.is_file()}
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                instrumentation="counters" if counters else "sampled",
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                source_sha256=sha(src / "prefill.csl"),
                files=files,
                scope="Original resident score_matmul then softmax_score, with explicit maximum initialization repair to -65504. Original communication and arithmetic retained, capture logits, partials and normalization scalars. Remove unrelated public roots. Source semantic probe, not HLS or complete attention qualification.",
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
    p.add_argument("--large", action="store_true")
    p.add_argument("--counters", action="store_true")
    a = p.parse_args()
    if a.prepare:
        prepare(128, 256, 8, a.counters) if a.large else prepare(counters=a.counters)
    elif a.execute:
        execute(a.execute.resolve(), 1800)
    else:
        mesh_half_worker(a.worker.resolve())
