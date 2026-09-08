"""Execute original Decode RMS and explicit input-DSD and accumulator repairs, side by side.

This is a source semantic probe, not HLS or full Decode qualification. Original
communication and numerical bodies remain visible in a diff; no host arithmetic
is injected between device stages.
"""

import argparse
import datetime
import difflib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from probe_runtime import execute, mesh_half_worker, read, sha, verify

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    source = ROOT / "projects/waferllm/upstream/Decode/src"
    root = (
        ROOT
        / "evidence"
        / (
            "decode-rms-source-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    shutil.copytree(source / "comm_lib", root / "comm_lib")
    shutil.copyfile(
        ROOT / "toolchain/runtime/batched_rms_local.csl", root / "batched_rms_local.csl"
    )
    original = (source / "decode.csl").read_text()
    start = original.index("fn rmsnorm_x() void {")
    end = original.index("\nfn xq_matvec_mult()", start)
    body = original[start:end]
    needle = "dim_p_pe_dsd_2 = @set_dsd_base_addr(dim_p_pe_dsd_2, ptr_X_tmp);"
    assert body.count(needle) == 1
    repaired = body.replace("fn rmsnorm_x()", "fn probe_repaired_rms()").replace(
        needle, needle.replace("ptr_X_tmp", "ptr_X")
    )
    reduce_start = repaired.index(
        "    dim_p_pe_dsd_1 = @set_dsd_base_addr(dim_p_pe_dsd_1, ptr_X_tmp);"
    )
    reduce_end = repaired.index("    comm_mod.all_reduce_bsz", reduce_start)
    repaired = repaired[:reduce_start] + """
    const acc=@get_dsd(mem1d_dsd,.{.base_address=&probe_accumulator,.extent=1});
    const element=@get_dsd(mem1d_dsd,.{.base_address=ptr_X_tmp,.extent=1});
    @load_to_dsr(dest_dsr_2,acc,.{.save_address=false});
    @load_to_dsr(src0_dsr_2,acc,.{.save_address=false});
    @load_to_dsr(src1_dsr_1,element,.{.save_address=true});
    for(@range(i16,bsz)) |b| {
        @fmovh(dest_dsr_2,0.0);
        for(@range(i16,dim_p_pe)) |j| {@faddh(dest_dsr_2,src0_dsr_2,src1_dsr_1);}
        local_sum[b]=probe_accumulator[0];
    }
""" + repaired[reduce_end:]
    call = "    comm_mod.all_reduce_bsz(py, quotient_y, remainder_y, ptr_local_sum);"

    def instrument(text, prefix):
        assert text.count(call) == 1
        before = f"@mov16(@get_dsd(mem1d_dsd,.{{.base_address=&{prefix}_before,.extent=bsz}}),local_sum_dsd);"
        after = f"@mov16(@get_dsd(mem1d_dsd,.{{.base_address=&{prefix}_after,.extent=bsz}}),local_sum_dsd);"
        return text.replace(call, before + "\n" + call + "\n" + after)

    observed_original = original[:start] + instrument(body, "original") + original[end:]
    repaired = instrument(repaired, "repaired")
    wrapper = """
const batched_rms=@import_module("batched_rms_local.csl",.{.batches=bsz,.features=dim_p_pe,.global_features=P*dim_p_pe,.epsilon=eps});
var library_result=@zeros([bsz*dim_p_pe]f16);
var library_sum=@zeros([bsz]f16);
var library_scratch=@zeros([bsz*dim_p_pe]f16);
var probe_accumulator=@zeros([1]f16);
var original_before=@zeros([bsz]f16);var original_after=@zeros([bsz]f16);
var repaired_before=@zeros([bsz]f16);var repaired_after=@zeros([bsz]f16);
var original_result=@zeros([bsz*dim_p_pe]f16);
var repaired_result=@zeros([bsz*dim_p_pe]f16);
var original_sum=@zeros([bsz]f16);
var repaired_sum=@zeros([bsz]f16);
var exp_arguments=[8]f16{0.0,-0.125,-0.5,-1.0,-2.0,-4.0,-8.0,-12.0};
var source_exp=@zeros([8]f16);
var progress=@zeros([1]u16);
fn probe() void {
 comm_mod.reconfig_allreduce_axis(1);
 rmsnorm_x();
 for(@range(i16,bsz*dim_p_pe)) |i| {original_result[i]=X_norm_tile[i];}
 for(@range(i16,bsz)) |i| {original_sum[i]=local_sum[i];}
 // Exercise phase routing reset before reusing the same queues and descriptors.
 comm_mod.reconfig_allreduce_axis(0);
 comm_mod.reconfig_allreduce_axis(1);
 probe_repaired_rms();
 for(@range(i16,bsz*dim_p_pe)) |i| {repaired_result[i]=X_norm_tile[i];}
 for(@range(i16,bsz)) |i| {repaired_sum[i]=local_sum[i];}
 for(@range(i16,8)) |i| {source_exp[i]=fast_exp(exp_arguments[i]);}
 batched_rms.square_sum(ptr_X,&library_scratch,&library_sum);
 comm_mod.all_reduce_bsz(py,quotient_y,remainder_y,&library_sum);
 batched_rms.normalize(ptr_X,ptr_W,&library_result,&library_sum);
 progress[0]+=1;
 sys_mod.unblock_cmd_stream();
}
comptime {@export_symbol(probe);}
"""
    ports = dict(
        library_result=16,
        library_sum=2,
        original_before=2,
        original_after=2,
        repaired_before=2,
        repaired_after=2,
        original_result=16,
        repaired_result=16,
        original_sum=2,
        repaired_sum=2,
        source_exp=8,
        progress=1,
    )
    for name in ports:
        dtype = "u16" if name == "progress" else "f16"
        wrapper += f'var ptr_probe_{name}:[*]{dtype}=&{name};comptime {{@export_symbol(ptr_probe_{name},"{name}");}}\n'
    adapted = observed_original + "\n" + repaired + wrapper
    (root / "decode.csl").write_text(adapted)
    layout = (source / "layout.csl").read_text()
    pos = layout.rfind("}")
    exports = "".join(
        f'@export_name("{name}",[*]{"u16" if name == "progress" else "f16"},true);\n'
        for name in ports
    )
    (root / "layout.csl").write_text(
        layout[:pos] + exports + '@export_name("probe",fn()void);\n' + layout[pos:]
    )
    (root / "source-adapter.diff").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                adapted.splitlines(True),
                fromfile="pinned/Decode/decode.csl",
                tofile="probe/decode.csl",
            )
        )
    )
    rng = np.random.default_rng(210107)
    xs = [rng.uniform(-0.5, 0.5, (2, 64)).astype(np.float16) for _ in range(8)]
    xs[1] = np.tile(np.array([-0.5, 0.25, 0.5, -0.25], np.float16), (2, 16))
    xs[2].fill(0)
    xs[3] = -np.abs(xs[3])
    xs[4][:, ::2] = 0
    logical = []
    physical = []
    for i, x in enumerate(xs):
        gamma = (np.ones(64) if i < 4 else rng.uniform(0.5, 1.5, 64)).astype(np.float16)
        logical.append(dict(x=x.tolist(), gamma=gamma.tolist()))
        # Feature shards on y; replicate each shard across x, batch-major locally.
        xx = np.empty((8, 8, 16), np.float16)
        ww = np.empty((8, 8, 8), np.float16)
        for y in range(8):
            xx[y, :, :] = x[:, y * 8 : (y + 1) * 8].ravel()
            ww[y, :, :] = gamma[y * 8 : (y + 1) * 8]
        physical.append(dict(X=xx.tolist(), W=ww.tolist()))
    files = {
        "logical-inputs.json": logical,
        "inputs.json": physical,
        "runtime-options.json": dict(
            suppress_trace=True, num_threads=8, dump_core=True
        ),
        "schema.json": dict(
            rows=8,
            cols=8,
            inputs=dict(X=16, W=8),
            outputs=dict(X=16, W=8, **ports),
            immutable=["X", "W"],
            progress="progress",
            initialize="init_task",
            launch="probe",
        ),
        "sdk-command.json": [
            "cslc",
            "layout.csl",
            "--arch=wse3",
            "--fabric-dims=15,10",
            "--fabric-offsets=4,1",
            "--params=P:8,bsz:2,dim_p_pe:8,pes_p_head:8,pes_p_kv_head:8,head_dim_p_pe:8,seq_len_p_pe:8,ffn_dim_p_pe:32,pe_num_p_group:4,root_1st_phase:2,root_2nd_phase:6",
            "-o=out",
            "--memcpy",
            "--channels=1",
        ],
    }
    for name, value in files.items():
        (root / name).write_text(json.dumps(value, indent=2) + "\n")
    shutil.copyfile(__file__, root / "driver.py")
    shutil.copyfile(ROOT / "experiments/probe_runtime.py", root / "probe_runtime.py")
    shutil.copyfile(ROOT / "toolchain/sdk_process.py", root / "sdk_process.py")
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                source_files={
                    str(p.relative_to(source)): sha(p) for p in source.rglob("*.csl")
                },
                adaptation=__doc__,
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


def analyze(root):
    verify(root)
    data = read(root / "results.json")
    inputs = read(root / "logical-inputs.json")
    assert data["success"] and len(data["cases"]) == len(inputs) == 8
    cases = []
    for epoch, (b, row) in enumerate(zip(inputs, data["cases"])):
        x = np.asarray(b["x"], float)
        g = np.asarray(b["gamma"], float)
        wanted = x * g / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-6)
        measures = {}
        for name in ("original_result", "repaired_result"):
            shards = np.asarray(row[name], np.uint16).view(np.float16).astype(float)
            for column in range(1, 8):
                np.testing.assert_array_equal(shards[:, column, :], shards[:, 0, :])
            actual = shards[:, 0, :].reshape(8, 2, 8).transpose(1, 0, 2).reshape(2, 64)
            l2 = float(
                np.linalg.norm(actual - wanted) / max(np.linalg.norm(wanted), 1e-30)
            )
            peak = float(
                np.max(np.abs(actual - wanted)) / max(np.max(np.abs(wanted)), 1e-30)
            )
            measures[name] = dict(
                relative_l2=l2,
                peak_scaled_error=peak,
                fixed_accuracy_passed=bool(
                    np.all(np.isfinite(actual)) and l2 <= 0.01 and peak <= 0.015
                ),
            )
        sums = (
            np.asarray(row["repaired_after"], np.uint16).view(np.float16).astype(float)
        )
        np.testing.assert_allclose(
            sums,
            np.broadcast_to(np.sum(x * x, axis=1), sums.shape),
            rtol=0.01,
            atol=2**-20,
        )
        before = np.asarray(row["original_before"], np.uint16).view(np.float16)
        last_squares = (x[0, 7::8].astype(np.float16) ** 2).astype(np.float16)
        np.testing.assert_array_equal(
            before[:, :, 0], np.broadcast_to(last_squares[:, None], (8, 8))
        )
        np.testing.assert_array_equal(row["library_result"], row["repaired_result"])
        np.testing.assert_array_equal(row["library_sum"], row["repaired_after"])
        assert measures["repaired_result"]["fixed_accuracy_passed"], (epoch, measures)
        cases.append(dict(epoch=epoch, **measures))
    assert any(not c["original_result"]["fixed_accuracy_passed"] for c in cases)
    expected = np.exp(np.array([0, -0.125, -0.5, -1, -2, -4, -8, -12], float))
    actual = (
        np.asarray(data["cases"][0]["source_exp"], np.uint16)
        .view(np.float16)
        .astype(float)
    )
    report = dict(
        passed=True,
        scope="Source discrepancy reproduction and explicit RMS input-DSD and memory-accumulator repair only; no HLS/full Decode qualification.",
        cases=cases,
        source_exp_samples=actual[0, 0].tolist(),
        mathematical_exp=expected.tolist(),
        source_exp_max_relative_error=float(
            np.max(np.abs(actual - expected) / expected)
        ),
        results_sha256=sha(root / "results.json"),
        provenance_sha256=sha(root / "provenance.json"),
        analysis_driver_sha256=sha(Path(__file__)),
    )
    out = root / "review.json"
    assert not out.exists()
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--worker", type=Path)
    p.add_argument("--execute", type=Path)
    p.add_argument("--analyze", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare()
    elif a.worker:
        mesh_half_worker(a.worker.resolve())
    elif a.execute:
        execute(a.execute.resolve(), 1200)
    elif a.analyze:
        analyze(a.analyze.resolve())
