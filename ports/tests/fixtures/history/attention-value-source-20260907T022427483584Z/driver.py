"""Probe source probability-times-V layout expectations before resident composition."""

import argparse, datetime, difflib, json, re, shutil, sys
from pathlib import Path
from probe_runtime import execute, mesh_half_worker, read, sha

ROOT = Path(__file__).resolve().parents[1]


def prepare(device_layout=False):
    sys.path.insert(0, str(ROOT / "toolchain"))
    import numpy as np
    from mesh_common import pack_tiles
    from mesh_twohop import block_index

    root = (
        ROOT
        / "evidence"
        / (
            "attention-value-source-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    src = ROOT / "projects/waferllm/upstream/Prefill/src"
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
        first = text.index("fn matmul_compute()")
        last = text.index("fn rmsnorm_x()", first)
        body = text[first:last]
        needle = "right_matrix_dsd = @increment_dsd_offset(right_matrix_dsd, Nt, f16);"
        assert body.count(needle) == 1
        body = body.replace(
            needle,
            "right_matrix_dsd = @increment_dsd_offset(right_matrix_dsd, 1, f16);",
        )
        text = text[:first] + body + text[last:]
        first = text.index("fn output_matmul()")
        last = text.index("fn h1_matmul()", first)
        body = text[first:last]
        needle = "    in_preshift = true;\n    pre_remaining = offset_step;\n    shift_round = 0;\n    left_matrix_shift_callback();"
        assert body.count(needle) == 1
        body = body.replace(
            needle,
            """    right_matrix_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{dim_p_pe}->dummy[i*seq_len_p_pe]});
    value_remaining=if(px==0) 0 else if(px%2==0) P-px/2 else (px+1)/2;
    value_step=0;value_phase=true;value_shift();""",
        )
        text = text[:first] + body + text[last:]
        needle = (
            "task right_matrix_finish() void {\n    @block(right_matrix_finish_id);"
        )
        assert text.count(needle) == 1
        text = text.replace(
            needle, needle + "\n    if(value_phase){value_shift();return;}"
        )
        text += """
var value_phase:bool=false;var value_remaining:i16=0;var value_step:i16=0;
fn value_shift() void {
 if(value_remaining>0){value_remaining-=1;swap_ptr=ptr_right_matrix_send;ptr_right_matrix_send=ptr_right_matrix_recv;ptr_right_matrix_recv=swap_ptr;comm_mod.mm_two_hop_comm_T(ptr_right_matrix_send,ptr_right_matrix_recv,value_step);value_step+=1;}
 else {value_phase=false;in_preshift=true;pre_remaining=offset_step;shift_round=0;left_matrix_shift_callback();}
}
"""
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
    base = ROOT / "evidence/score-matmul-source-20260907T015139514873Z"
    for name in ["sdk-command.json", "runtime-options.json", "WaferLLM-LICENSE.txt"]:
        shutil.copyfile(base / name, root / name)
    for name, source in [
        ("driver.py", Path(__file__)),
        ("probe_runtime.py", ROOT / "experiments/probe_runtime.py"),
        ("sdk_process.py", ROOT / "toolchain/sdk_process.py"),
    ]:
        shutil.copyfile(source, root / name)
    m, n, p = 64, 128, 8
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
                outputs=dict(output=mt * nt, history=p * mt * nt, timing=6, progress=1),
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
    a = p.parse_args()
    if a.prepare:
        prepare(a.device_layout)
    elif a.execute:
        execute(a.execute.resolve(), 1800)
    else:
        mesh_half_worker(a.worker.resolve())
