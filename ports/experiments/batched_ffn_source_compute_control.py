"""Pinned Decode vecmat control under the same explicit SDK collective schedule.

This isolates local projection lowering overhead. It does not compare against
unmodified full Decode or claim source RMS/collective/SiLU implementation parity.
"""

import argparse, datetime, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from probe_runtime import execute, mesh_half_worker, sha, read, verify


def prepare(bundle):
    sys.path.insert(0, str(bundle / "implementation"))
    from integrity import verify_bundle
    from mesh_batched_feed_forward_sdk import packed, extents, parameters

    verify_bundle(bundle)
    s, m, batches = [
        read(bundle / n) for n in ("schedule.json", "semantic.json", "batches.json")
    ]
    root = (
        ROOT
        / "evidence"
        / (
            "batched-ffn-source-compute-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    for p in bundle.glob("*.csl"):
        shutil.copyfile(p, root / p.name)
    source = ROOT / "projects/waferllm/upstream/Decode/src/decode.csl"
    text = source.read_text()

    def function(name):
        start = text.index("fn " + name + "(")
        a = text.index("{", start)
        depth = 1
        end = a + 1
        while depth:
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            end += 1
        return text[start:end]

    original_functions = {
        name: function(name) for name in ("gemv_static_step", "vecmat_computation")
    }
    code = (
        """// Original WaferLLM Decode arithmetic; only module/caller interface adapted.
param batches:i16;param inner:i16;param columns:i16;
const bsz:i16=batches;
const dest_dsr_1=@get_dsr(dsr_dest,1);const src0_dsr_1=@get_dsr(dsr_src0,1);const src1_dsr_1=@get_dsr(dsr_src1,1);
var dummy=@zeros([1]f16);
var left_vector_dsd=@get_dsd(mem1d_dsd,.{.base_address=&dummy,.extent=1});
var right_matrix_dsd=@get_dsd(mem1d_dsd,.{.base_address=&dummy,.extent=1});
var out_vector_dsd=@get_dsd(mem1d_dsd,.{.base_address=&dummy,.extent=1});
var ptr_right_matrix:[*]f16=&dummy;
var Kt:i16=0;var Nt:i16=0;
"""
        + "\n".join(original_functions.values())
        + """
fn compute(input:[*]f16,weight:[*]f16,output:[*]f16) void {
 Kt=inner;Nt=columns;
 const all_output=@get_dsd(mem1d_dsd,.{.base_address=output,.extent=@as(u16,batches*columns)});
 @fmovh(all_output,0.0);
 left_vector_dsd=@set_dsd_base_addr(left_vector_dsd,input);
 ptr_right_matrix=weight;
 out_vector_dsd=@set_dsd_base_addr(out_vector_dsd,output);
 vecmat_computation();
}
"""
    )
    (root / "source_vecmat.csl").write_text(code)
    p = root / "pe.csl"
    p.write_text(
        p.read_text().replace('"batched_matmul_local.csl"', '"source_vecmat.csl"')
    )
    inputs = [{k: v.tolist() for k, v in packed(s, m, b).items()} for b in batches]
    files = {
        "inputs.json": inputs,
        "schema.json": dict(
            rows=s["P"],
            cols=s["P"],
            inputs={k: len(v[0][0]) for k, v in inputs[0].items()},
            outputs=extents(s),
            immutable=["X", "gamma", "weights"],
            initialize="init_task",
            launch="hls_main",
        ),
        "runtime-options.json": read(bundle / "runtime-options.json"),
        "sdk-command.json": [
            "cslc",
            "layout.csl",
            "--arch=wse3",
            f'--fabric-dims={s["P"]+7},{s["P"]+2}',
            "--fabric-offsets=4,1",
            parameters(s),
            "-o=out",
            "--memcpy",
            "--channels=1",
        ],
        "source-functions.json": original_functions,
    }
    for name, value in files.items():
        (root / name).write_text(json.dumps(value) + "\n")
    for p, n in [
        (Path(__file__), "driver.py"),
        (ROOT / "experiments/probe_runtime.py", "probe_runtime.py"),
        (ROOT / "toolchain/sdk_process.py", "sdk_process.py"),
    ]:
        shutil.copyfile(p, root / n)
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                hls_bundle=str(bundle.relative_to(ROOT)),
                hls_manifest_sha256=sha(bundle / "manifest.json"),
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                source_sha256=sha(source),
                scope="Unchanged original vecmat functions; same HLS SDK planes, RMS repair, stable SiLU and instrumentation. Local projection lowering control only, not original full Decode performance.",
                files={p.name: sha(p) for p in root.iterdir() if p.is_file()},
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", type=Path)
    p.add_argument("--worker", type=Path)
    p.add_argument("--execute", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.prepare.resolve())
    elif a.worker:
        mesh_half_worker(a.worker.resolve())
    elif a.execute:
        execute(a.execute.resolve(), 4000)
