"""Execute source rectangular up/gate/SiLU/down with explicit branch-owner control."""

import argparse, datetime, difflib, json, re, shutil, sys
from pathlib import Path
from probe_runtime import execute, mesh_half_worker, read, sha

ROOT = Path(__file__).resolve().parents[1]


def prepare(repair=False, sampled=False, m=64, n=64, f=256, p=8, blocked=False):
    assert p in (4, 8) and all(
        type(v) is int and 1 <= v <= 512 and v % p == 0 for v in (m, n, f)
    )
    mt, nt, ft = m // p, n // p, f // p
    sys.path.insert(0, str(ROOT / "toolchain"))
    import numpy as np
    from mesh_common import pack_tiles
    from mesh_twohop import block_index

    root = (
        ROOT
        / "evidence"
        / (
            "mlp-source-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    src = ROOT / "projects/waferllm/upstream/Prefill/src"
    original = (src / "prefill.csl").read_text()
    text = original
    start = text.index("fn prefill_struct() void {")
    end = text.index(
        "\n// --------------------------------------------------------------------------",
        start,
    )
    text = text[:start] + """fn prefill_struct() void {
 if(hls_phase==0){hls_phase=1;z2_matmul();}
 else if(hls_phase==1){if(HLS_SAMPLED){@fmovh(hls_gate_dsd,z2_dsd);}hls_phase=2;z3_comp();}
 else if(hls_phase==2){if(HLS_SAMPLED){@fmovh(hls_hidden_dsd,z3_dsd);}hls_phase=3;h2_matmul();}
 else {hls_finish();}
}
""" + text[end:]
    text = re.sub(r"@export_symbol\([^;]*?\);", "", text)
    if repair:
        a = text.index("fn z2_matmul()")
        b = text.index("fn silu_kernel", a)
        body = text[a:b]
        needle = "    ptr_left_matrix_send = &seqLen_dim_tmp;\n    ptr_left_matrix_recv = &Z_norm_tile;"
        assert body.count(needle) == 1
        body = body.replace(
            needle,
            "    swap_ptr=ptr_left_matrix_send;\n    ptr_left_matrix_send=ptr_left_matrix_recv;\n    ptr_left_matrix_recv=swap_ptr;",
        )
        text = text[:a] + body + text[b:]
    a = text.index("fn matmul_compute()")
    b = text.index("fn rmsnorm_x()", a)
    body = text[a:b]
    needle = "        step += 1;"
    assert body.count(needle) == 1
    body = body.replace(
        needle,
        """        if(HLS_SAMPLED){
         if(hls_phase==0){@fmovh(@increment_dsd_offset(hls_uh,step*seq_len_p_pe*ffn_dim_p_pe,f16),z1_dsd);}
         else if(hls_phase==1){@fmovh(@increment_dsd_offset(hls_gh,step*seq_len_p_pe*ffn_dim_p_pe,f16),z2_dsd);if(step==0){@fmovh(hls_owner_dsd,@set_dsd_base_addr(hls_owner_dsd,ptr_left_matrix_send));}}
         else {@fmovh(@increment_dsd_offset(hls_dh,step*seq_len_p_pe*dim_p_pe,f16),h2_dsd);}
        }
""" + needle,
    )
    if blocked:
        assert body.count("    if (step < P) {") == 1
        body = body.replace(
            "    if (step < P) {",
            "    if (step < P) {\n        if(hls_phase==3){@fmovh(h2_dsd,0.0);}",
        )
        body = body.replace(
            "        if(HLS_SAMPLED){",
            "        if(hls_phase==3){block_accum.merge(ptr_h2); }\n        if(HLS_SAMPLED){",
        )
    text = text[:a] + body + text[b:]
    text += f"\nconst HLS_SAMPLED:bool={str(sampled).lower()};\n"
    text += """
var hls_phase:i16=0;var hls_gate=@zeros([if(HLS_SAMPLED) seq_len_p_pe*ffn_dim_p_pe else 1]f16);var hls_hidden=@zeros([if(HLS_SAMPLED) seq_len_p_pe*ffn_dim_p_pe else 1]f16);
var hls_up_history=@zeros([if(HLS_SAMPLED) P*seq_len_p_pe*ffn_dim_p_pe else 1]f16);var hls_gate_history=@zeros([if(HLS_SAMPLED) P*seq_len_p_pe*ffn_dim_p_pe else 1]f16);var hls_down_history=@zeros([if(HLS_SAMPLED) P*seq_len_p_pe*dim_p_pe else 1]f16);var hls_owner=@zeros([if(HLS_SAMPLED) seq_len_p_pe*dim_p_pe else 1]f16);
const hls_gate_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*ffn_dim_p_pe}->hls_gate[i]});const hls_hidden_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*ffn_dim_p_pe}->hls_hidden[i]});
const hls_uh=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*ffn_dim_p_pe}->hls_up_history[i]});const hls_gh=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*ffn_dim_p_pe}->hls_gate_history[i]});const hls_dh=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*dim_p_pe}->hls_down_history[i]});const hls_owner_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe*dim_p_pe}->hls_owner[i]});
var hls_start=@zeros([3]u16);var hls_end=@zeros([3]u16);var hls_timing=@zeros([6]u16);var hls_progress=@zeros([1]u16);
fn hls_main() void {hls_phase=0;timestamp.enable_tsc();timestamp.get_timestamp(&hls_start);z1_matmul();}
fn hls_finish() void {timestamp.get_timestamp(&hls_end);timestamp.disable_tsc();for(@range(i16,3)) |i| {hls_timing[i]=hls_start[i];hls_timing[i+3]=hls_end[i];}hls_progress[0]+=1;sys_mod.unblock_cmd_stream();}
var hls_gp:[*]f16=&hls_gate;var hls_hp:[*]f16=&hls_hidden;var hls_uhp:[*]f16=&hls_up_history;var hls_ghp:[*]f16=&hls_gate_history;var hls_dhp:[*]f16=&hls_down_history;var hls_op:[*]f16=&hls_owner;var hls_tp:[*]u16=&hls_timing;var hls_pp:[*]u16=&hls_progress;
comptime {@export_symbol(ptr_Z_norm,"x");@export_symbol(ptr_UP_weight,"up_weight");@export_symbol(ptr_GATE_weight,"gate_weight");@export_symbol(ptr_DOWN_weight,"down_weight");@export_symbol(ptr_z1,"up");@export_symbol(ptr_z2,"activated_gate");@export_symbol(ptr_h2,"output");@export_symbol(hls_gp,"gate");@export_symbol(hls_hp,"hidden");@export_symbol(hls_uhp,"up_history");@export_symbol(hls_ghp,"gate_history");@export_symbol(hls_dhp,"down_history");@export_symbol(hls_op,"gate_left_owner");@export_symbol(hls_tp,"timing");@export_symbol(hls_pp,"progress");@export_symbol(init_task);@export_symbol(hls_main);}
"""
    if blocked:
        text = text.replace(
            "fn h2_matmul() void {", "fn h2_matmul() void {\n    block_accum.reset();"
        )
        text = text.replace(
            "fn hls_finish() void {timestamp",
            "fn hls_finish() void {block_accum.snapshot();timestamp",
        )
        text += '\nconst block_accum=@import_module("block_accumulate.csl",.{.length=seq_len_p_pe*dim_p_pe});\nvar wide_ptr:[*]u32=block_accum.words;\ncomptime {@export_symbol(wide_ptr,"wide_accumulator");}\n'
        shutil.copyfile(
            ROOT / "toolchain/runtime/block_accumulate.csl",
            root / "block_accumulate.csl",
        )
    (root / "prefill.csl").write_text(text)
    (root / "source-adapter.diff").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                text.splitlines(True),
                fromfile="pinned/prefill.csl",
                tofile="mlp/prefill.csl",
            )
        )
    )
    layout = re.sub(r"@export_name\([^;]*?\);", "", (src / "layout.csl").read_text())
    pos = layout.rfind("}")
    names = [
        "x",
        "up_weight",
        "gate_weight",
        "down_weight",
        "up",
        "activated_gate",
        "output",
        "gate",
        "hidden",
        "up_history",
        "gate_history",
        "down_history",
        "gate_left_owner",
    ]
    exports = (
        "".join(f'@export_name("{name}",[*]f16,true);' for name in names)
        + '@export_name("timing",[*]u16,true);@export_name("progress",[*]u16,true);@export_name("init_task",fn()void);@export_name("hls_main",fn()void);\n'
    )
    if blocked:
        exports += '@export_name("wide_accumulator",[*]u32,true);\n'
    (root / "layout.csl").write_text(layout[:pos] + exports + layout[pos:])
    shutil.copytree(src / "comm_lib", root / "comm_lib")
    base = ROOT / "evidence/resident-attention-source-20260907T032018473963Z"
    for name in ("sdk-command.json", "runtime-options.json", "WaferLLM-LICENSE.txt"):
        shutil.copyfile(base / name, root / name)
    cmd = [
        (
            f"--params=P:{p},dim_p_pe:{nt},pes_p_head:{p},pes_p_kv_head:{p},head_dim_p_pe:{nt},seq_len_p_pe:{mt},ffn_dim_p_pe:{ft}"
            if a.startswith("--params=")
            else f"--fabric-dims={p+7},{p+2}" if a.startswith("--fabric-dims=") else a
        )
        for a in read(root / "sdk-command.json")
    ]
    (root / "sdk-command.json").write_text(json.dumps(cmd) + "\n")
    for name, source in [
        ("driver.py", Path(__file__)),
        ("probe_runtime.py", ROOT / "experiments/probe_runtime.py"),
        ("sdk_process.py", ROOT / "toolchain/sdk_process.py"),
    ]:
        shutil.copyfile(source, root / name)
    rng = np.random.default_rng(210110)
    logical = []
    bs = []

    def weights(a):
        kt, ot = a.shape[0] // p, a.shape[1] // p
        return [
            [
                a[
                    block_index(p, y, x) * kt : (block_index(p, y, x) + 1) * kt,
                    x * ot : (x + 1) * ot,
                ]
                .ravel(order="C")
                .tolist()
                for x in range(p)
            ]
            for y in range(p)
        ]

    for epoch in range(6 if blocked else 3):
        x = np.asarray(rng.uniform(-0.125, 0.125, (m, n)), np.float16).astype(float)
        if epoch == 2:
            x.fill(0)
        u, g, d = [
            np.asarray(rng.uniform(-0.125, 0.125, shape), np.float16).astype(float)
            for shape in ((n, f), (n, f), (f, n))
        ]
        if blocked:
            if epoch == 3:
                x.fill(0.125)
                g.fill(-0.125)
            if epoch == 4:
                d.fill(0)
            if epoch == 5:
                for v in (x, u, g, d):
                    v.fill(0.125)
        logical.append(
            {
                name: a.ravel().tolist()
                for name, a in zip(
                    ("x", "up_weight", "gate_weight", "down_weight"), (x, u, g, d)
                )
            }
        )
        bs.append(
            dict(
                x=pack_tiles(x, p, p, "F").tolist(),
                up_weight=weights(u),
                gate_weight=weights(g),
                down_weight=weights(d),
            )
        )
    if blocked:
        x = np.full((m, n), 0.125)
        u = np.full((n, f), 0.125)
        g = u.copy()
        d = np.zeros((f, n))
        for pair in range(p // 2 - 1):
            d[2 * pair * ft : (2 * pair + 1) * ft] = 0.125 / (2**pair)
            d[(2 * pair + 1) * ft : (2 * pair + 2) * ft] = -0.125 / (2**pair)
        d[(p - 2) * ft : (p - 1) * ft] = 2**-10
        extra = [
            (x, u, g, d),
            (
                rng.uniform(-0.125, 0.125, (m, n)),
                rng.uniform(-0.125, 0.125, (n, f)),
                rng.uniform(-0.125, 0.125, (n, f)),
                np.zeros((f, n)),
            ),
        ]
        for values in extra:
            x, u, g, d = [np.asarray(v, np.float16).astype(float) for v in values]
            logical.append(
                {
                    name: v.ravel().tolist()
                    for name, v in zip(
                        ("x", "up_weight", "gate_weight", "down_weight"), (x, u, g, d)
                    )
                }
            )
            bs.append(
                dict(
                    x=pack_tiles(x, p, p, "F").tolist(),
                    up_weight=weights(u),
                    gate_weight=weights(g),
                    down_weight=weights(d),
                )
            )
    schema = dict(
        rows=p,
        cols=p,
        inputs=dict(
            x=mt * nt, up_weight=nt * ft, gate_weight=nt * ft, down_weight=nt * ft
        ),
        outputs=dict(
            up=mt * ft,
            activated_gate=mt * ft,
            output=mt * nt,
            gate=mt * ft if sampled else 1,
            hidden=mt * ft if sampled else 1,
            up_history=p * mt * ft if sampled else 1,
            gate_history=p * mt * ft if sampled else 1,
            down_history=p * mt * nt if sampled else 1,
            gate_left_owner=mt * nt if sampled else 1,
            timing=6,
            progress=1,
        ),
        launch="hls_main",
        initialize="init_task",
        progress="progress",
    )
    if blocked:
        schema["outputs"]["wide_accumulator"] = mt * nt
        schema["output_word_bits"] = {"wide_accumulator": 32}
    for name, data in [
        ("inputs.json", bs),
        ("logical-inputs.json", logical),
        ("schema.json", schema),
        ("geometry.json", dict(M=m, N=n, F=f, P=p)),
    ]:
        (root / name).write_text(json.dumps(data) + "\n")
    files = {str(v.relative_to(root)): sha(v) for v in root.rglob("*") if v.is_file()}
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                blocked_down_accumulation=blocked,
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                source_sha256=sha(src / "prefill.csl"),
                preserve_completed_left_ownership=repair,
                instrumentation="sampled" if sampled else "counters",
                files=files,
                scope="Source-only supplied normalized activation -> rectangular up/gate projections -> source SiLU/product -> down projection. Static weights host-packed into source-prescribed block ownership. No intermediate host transfers. Optional explicit gate-branch live-pointer repair, otherwise original source. Not HLS, RMS/residual or full prefill/decode qualification.",
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
    p.add_argument("--repair", action="store_true")
    p.add_argument("--sampled", action="store_true")
    p.add_argument("--large", action="store_true")
    p.add_argument("--geometry", type=int, nargs=4, metavar=("M", "N", "F", "P"))
    p.add_argument("--blocked-down", action="store_true")
    a = p.parse_args()
    assert not (a.large and a.geometry)
    if a.prepare and a.geometry:
        prepare(a.repair, a.sampled, *a.geometry, blocked=a.blocked_down)
    elif a.prepare:
        (
            prepare(a.repair, a.sampled, 128, 128, 512, blocked=a.blocked_down)
            if a.large
            else prepare(a.repair, a.sampled, blocked=a.blocked_down)
        )
    elif a.execute:
        execute(a.execute.resolve(), 2400)
    else:
        mesh_half_worker(a.worker.resolve())
