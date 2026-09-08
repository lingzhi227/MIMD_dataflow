"""Source h1 projection/residual/RMS: isolate descriptor and row-scaling repairs.

The harness binds the stale RMS input descriptor to a deterministic zero buffer.
This models prior-stage descriptor state with a valid extent; it is not an
unmodified full Prefill execution. Original numerical functions remain intact
except for the explicitly selected repairs and observation hooks.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, difflib, json, re, shutil
from pathlib import Path
import numpy as np
from probe_runtime import execute, mesh_half_worker, read, sha

ROOT = repository_root(__file__)


def prepare(repair, m=64, n=64, p=8):
    import sys

    sys.path.insert(0, str(ROOT / "lib"))
    from mesh_common import pack_tiles
    from mesh_twohop import block_index

    assert p in (4, 8) and 0 < m <= 128 and 0 < n <= 256
    assert m % p == 0 and n % p == 0
    assert (
        repair in ("both", "library") or m == n
    ), "original feature-index control requires square geometry"
    mt, nt = m // p, n // p
    root = (
        ROOT
        / "validation/evidence"
        / (
            "projection-residual-rms-source-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    src = ROOT / "third_party/sources/waferllm/Prefill/src"
    original = (src / "prefill.csl").read_text()
    text = re.sub(r"@export_symbol\([^;]*?\);", "", original)
    a = text.index("fn prefill_struct() void {")
    b = text.index(
        "\n// --------------------------------------------------------------------------",
        a,
    )
    text = text[:a] + """fn prefill_struct() void {
 if(hls_phase==0){hls_phase=1;z_add();}
 else if(hls_phase==1){hls_phase=2;rmsnorm_z();}
 else {hls_finish();}
}
""" + text[b:]
    a = text.index("fn rmsnorm_z() void {")
    b = text.index("\nfn z1_matmul()", a)
    body = text[a:b]
    wrong = "@load_to_dsr(comp_src1_dsr_2, seqLen_dsd_2, .{ .save_address = true });"
    assert body.count(wrong) == 1
    if repair in ("descriptor", "both"):
        body = body.replace(wrong, wrong.replace("seqLen_dsd_2", "seqLen_dsd_1"))
    needle = "    comm_mod.mv_allreduce_add_x(ptr_local_sum);"
    assert body.count(needle) == 1
    body = body.replace(
        needle,
        "@fmovh(hls_local_dsd,local_sum_dsd);\n"
        + needle
        + "\n@fmovh(hls_total_dsd,local_sum_dsd);",
    )
    wrong = """    for (@range(i16, dim_p_pe)) |i| {
        @fmulh(comp_dest_dsr_1, comp_src0_dsr_1, local_sum[i]);
    }"""
    assert body.count(wrong) == 1
    if repair == "both":
        body = body.replace(
            wrong,
            """    @load_to_dsr(comp_src1_dsr_1,local_sum_dsd,.{.save_address=false});
    for (@range(i16, dim_p_pe)) |i| {
        @fmulh(comp_dest_dsr_1, comp_src0_dsr_1, comp_src1_dsr_1);
    }""",
        )
    if repair == "library":
        body = """fn rmsnorm_z() void {
 rms_local.square_sum(ptr_Z,ptr_seqLen_dim_tmp,ptr_local_sum);
 @fmovh(hls_local_dsd,local_sum_dsd);
 comm_mod.mv_allreduce_add_x(ptr_local_sum);
 @fmovh(hls_total_dsd,local_sum_dsd);
 rms_local.inverse(ptr_local_sum);
 rms_local.normalize(ptr_Z,ptr_W,ptr_Z_norm,ptr_local_sum);
 prefill_struct();
}
"""
        shutil.copyfile(
            ROOT / "runtime/csl/rms_local.csl", root / "rms_local.csl"
        )
    text = text[:a] + body + text[b:]
    if repair == "library":
        text += '\nconst rms_local=@import_module("rms_local.csl",.{.rows=seq_len_p_pe,.features=dim_p_pe,.global_features=dim_p_pe*P,.epsilon=eps});\n'

    text += """
var hls_phase:i16=0;
var hls_stale=@zeros([seq_len_p_pe*dim_p_pe]f16);
var hls_local=@zeros([seq_len_p_pe]f16);var hls_total=@zeros([seq_len_p_pe]f16);
const hls_local_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe}->hls_local[i]});
const hls_total_dsd=@get_dsd(mem1d_dsd,.{.tensor_access=|i|{seq_len_p_pe}->hls_total[i]});
var hls_start=@zeros([3]u16);var hls_end=@zeros([3]u16);var hls_time=@zeros([6]u16);var hls_progress=@zeros([1]u16);
fn hls_main() void {
 hls_phase=0;
 seqLen_dsd_2=@set_dsd_base_addr(seqLen_dsd_2,&hls_stale);
 timestamp.enable_tsc();timestamp.get_timestamp(&hls_start);h1_matmul();
}
fn hls_finish() void {
 timestamp.get_timestamp(&hls_end);timestamp.disable_tsc();
 for(@range(i16,3)) |i| {hls_time[i]=hls_start[i];hls_time[i+3]=hls_end[i];}
 hls_progress[0]+=1;sys_mod.unblock_cmd_stream();
}
var hls_lp:[*]f16=&hls_local;var hls_rp:[*]f16=&hls_total;
var hls_tp:[*]u16=&hls_time;var hls_pp:[*]u16=&hls_progress;
comptime {
 @export_symbol(ptr_output,"activation");@export_symbol(ptr_O_weight,"weight");@export_symbol(ptr_X,"residual");@export_symbol(ptr_W,"gamma");
 @export_symbol(ptr_h1,"projection");@export_symbol(ptr_Z,"sum");@export_symbol(ptr_Z_norm,"result");
 @export_symbol(hls_lp,"local_square_sum");@export_symbol(hls_rp,"reduced_square_sum");@export_symbol(ptr_local_sum,"inverse");
 @export_symbol(hls_tp,"timing");@export_symbol(hls_pp,"progress");@export_symbol(init_task);@export_symbol(hls_main);
}
"""
    (root / "prefill.csl").write_text(text)
    (root / "source-adapter.diff").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                text.splitlines(True),
                fromfile="pinned/prefill.csl",
                tofile="probe/prefill.csl",
            )
        )
    )
    layout = re.sub(r"@export_name\([^;]*?\);", "", (src / "layout.csl").read_text())
    inputs = dict(activation=mt * nt, weight=nt * nt, residual=mt * nt, gamma=nt)
    outputs = dict(
        projection=mt * nt,
        sum=mt * nt,
        result=mt * nt,
        local_square_sum=mt,
        reduced_square_sum=mt,
        inverse=mt,
        timing=6,
        progress=1,
    )
    exports = "".join(
        f'@export_name("{k}",[*]{"u16" if k in ("timing","progress") else "f16"},true);'
        for k in inputs | outputs
    )
    exports += '@export_name("init_task",fn()void);@export_name("hls_main",fn()void);\n'
    at = layout.rfind("}")
    (root / "layout.csl").write_text(layout[:at] + exports + layout[at:])
    shutil.copytree(src / "comm_lib", root / "comm_lib")
    base = ROOT / "validation/evidence/mlp-source-20260907T055824962168Z"
    for name in ("runtime-options.json", "WaferLLM-LICENSE.txt"):
        shutil.copyfile(base / name, root / name)
    cmd = [
        (
            f"--params=P:{p},dim_p_pe:{nt},pes_p_head:{p},pes_p_kv_head:{p},head_dim_p_pe:{nt},seq_len_p_pe:{mt},ffn_dim_p_pe:{nt}"
            if v.startswith("--params=")
            else f"--fabric-dims={p+7},{p+2}" if v.startswith("--fabric-dims=") else v
        )
        for v in read(base / "sdk-command.json")
    ]
    rng = np.random.default_rng(210112)
    logical = []
    physical = []
    for epoch in range(3):
        activation = rng.uniform(-0.125, 0.125, (m, n)).astype(np.float16).astype(float)
        weight = rng.uniform(-0.125, 0.125, (n, n)).astype(np.float16).astype(float)
        residual = rng.uniform(-0.5, 0.5, (m, n)).astype(np.float16).astype(float)
        # Nonuniform rows and gamma distinguish descriptor and scale indexing.
        residual = (
            (residual * np.linspace(0.25, 1, m)[:, None])
            .astype(np.float16)
            .astype(float)
        )
        gamma = rng.uniform(0.5, 1.5, (1, n)).astype(np.float16).astype(float)
        if epoch == 2:
            activation.fill(0)
            residual.fill(0)
        logical.append(
            {
                k: v.ravel().tolist()
                for k, v in zip(inputs, (activation, weight, residual, gamma))
            }
        )
        wp = [
            [
                weight[
                    block_index(p, y, x) * nt : (block_index(p, y, x) + 1) * nt,
                    x * nt : (x + 1) * nt,
                ]
                .ravel(order="C")
                .tolist()
                for x in range(p)
            ]
            for y in range(p)
        ]
        gp = np.broadcast_to(gamma.reshape(p, nt)[None, :, :], (p, p, nt)).tolist()
        physical.append(
            dict(
                activation=pack_tiles(activation, p, p, "F").tolist(),
                weight=wp,
                residual=pack_tiles(residual, p, p, "F").tolist(),
                gamma=gp,
            )
        )
    for name, data in [
        ("inputs.json", physical),
        ("logical-inputs.json", logical),
        ("geometry.json", dict(M=m, N=n, P=p)),
        ("sdk-command.json", cmd),
        (
            "schema.json",
            dict(
                rows=p,
                cols=p,
                inputs=inputs,
                outputs=outputs,
                launch="hls_main",
                initialize="init_task",
                progress="progress",
            ),
        ),
    ]:
        (root / name).write_text(json.dumps(data) + "\n")
    for name, source in [
        ("driver.py", Path(__file__)),
        ("probe_runtime.py", ROOT / "experiments/probe_runtime.py"),
        ("sdk_process.py", ROOT / "lib/Runtime/sdk_process.py"),
    ]:
        shutil.copyfile(source, root / name)
    files = {str(v.relative_to(root)): sha(v) for v in root.rglob("*") if v.is_file()}
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                repair=repair,
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                source_sha256=sha(src / "prefill.csl"),
                files=files,
                scope=__doc__,
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
    p.add_argument(
        "--repair", choices=["none", "descriptor", "both", "library"], default="none"
    )
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--p", type=int, default=8)
    a = p.parse_args()
    if a.prepare:
        prepare(a.repair, a.m, a.n, a.p)
    elif a.execute:
        execute(a.execute.resolve(), 2400)
    else:
        mesh_half_worker(a.worker.resolve())
