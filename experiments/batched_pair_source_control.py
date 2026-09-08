"""Decode xq_rope body under matched batch-major HLS transport and observations."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, shutil, sys, difflib
from pathlib import Path
from probe_runtime import read, sha, execute, verify

ROOT = repository_root(__file__)


def prepare(hls):
    s = read(hls / "schedule.json")
    assert (
        s["layout"] == "batch_major"
        and s["pair_order"] == "odd_even"
        and s["broadcast_coefficients"]
    )
    root = (
        ROOT
        / "validation/evidence"
        / (
            "batched-pair-source-offset-repaired-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    source = ROOT / "third_party/sources/waferllm/Decode/src/decode.csl"
    text = source.read_text()
    original = text[text.index("fn xq_rope()") : text.index("fn xk_rope()")]
    needle = "        @load_to_dsr(dest_dsr_5, X_even_dsd);"
    assert original.count(needle) == 1
    sample = """        if(sampled!=0){
          @fmovh(@increment_dsd_offset(hd,b*2*dim_p_pe,f16),X_tmp_1_dsd);
          @fmovh(@increment_dsd_offset(hd,b*2*dim_p_pe+dim_p_pe/2,f16),X_tmp_2_dsd);
          @fmovh(@increment_dsd_offset(hd,b*2*dim_p_pe+dim_p_pe,f16),X_tmp_3_dsd);
          @fmovh(@increment_dsd_offset(hd,b*2*dim_p_pe+3*(dim_p_pe/2),f16),X_tmp_4_dsd);
        }
"""
    reset = "    X_odd_dsd = @set_dsd_base_addr(X_odd_dsd, ptr_QKV_tile);"
    assert original.count(reset) == 1
    body = original.replace(
        reset, reset + "\n    X_odd_dsd = @increment_dsd_offset(X_odd_dsd, 1, f16);"
    ).replace(needle, sample + needle)
    module = """param batches:i16;param features:i16;param sampled:i16;param swapped:i16;
const bsz:i16=batches;const dim_p_pe:i16=features;const _dim_p_pe:i16=features;
var dummy=@zeros([1]f16);var ptr_QKV_tile:[*]f16=&dummy;
var X_even_dsd=@get_dsd(mem1d_dsd,.{.base_address=&dummy,.offset=0,.extent=_dim_p_pe/2,.stride=2});
var X_odd_dsd=@get_dsd(mem1d_dsd,.{.base_address=&dummy,.offset=1,.extent=_dim_p_pe/2,.stride=2});
var freqs_cos_dsd=@get_dsd(mem1d_dsd,.{.base_address=&dummy,.extent=_dim_p_pe/2});
var freqs_sin_dsd=@get_dsd(mem1d_dsd,.{.base_address=&dummy,.extent=_dim_p_pe/2});
var hd=@get_dsd(mem1d_dsd,.{.base_address=&dummy,.extent=_dim_p_pe/2});
"""
    for kind, typ in [("dest", "dsr_dest"), ("src0", "dsr_src0"), ("src1", "dsr_src1")]:
        for k in range(1, 6):
            module += f"const {kind}_dsr_{k}=@get_dsr({typ},{k});\n"
    for k in range(1, 5):
        module += f"var X_tmp_{k}_dsd=@get_dsd(mem1d_dsd,.{{.base_address=&dummy,.extent=_dim_p_pe/2}});\n"
    module += (
        body
        + "\nfn apply(input:[*]f16,output:[*]f16,cosine:[*]f16,sine:[*]f16,scratch:[*]f16,history:[*]f16) void {\n"
    )
    module += "const xd=@get_dsd(mem1d_dsd,.{.base_address=input,.extent=batches*features});const yd=@set_dsd_base_addr(xd,output);@fmovh(yd,xd);ptr_QKV_tile=output;\n"
    module += "freqs_cos_dsd=@set_dsd_base_addr(freqs_cos_dsd,cosine);freqs_sin_dsd=@set_dsd_base_addr(freqs_sin_dsd,sine);hd=@set_dsd_base_addr(hd,history);\n"
    for k in range(1, 5):
        module += f"X_tmp_{k}_dsd=@set_dsd_base_addr(X_tmp_{k}_dsd,scratch);X_tmp_{k}_dsd=@increment_dsd_offset(X_tmp_{k}_dsd,{k-1}*(features/2),f16);\n"
    module += "xq_rope();\n}\n"
    for name in (
        "layout.csl",
        "pe.csl",
        "schedule.json",
        "semantic.json",
        "batches.json",
        "runtime-options.json",
        "WaferLLM-LICENSE.txt",
    ):
        shutil.copyfile(hls / name, root / name)
    (root / "batched_pair_rotation_local.csl").write_text(module)
    (root / "original-kernel.csl").write_text(original)
    (root / "source-body-observation.diff").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                body.splitlines(True),
                fromfile="original",
                tofile="observed",
            )
        )
    )
    shutil.copytree(hls / "implementation", root / "implementation")
    for path, name in [
        (Path(__file__), "driver.py"),
        (ROOT / "experiments/probe_runtime.py", "probe_runtime.py"),
        (ROOT / "lib/Runtime/sdk_process.py", "sdk_process.py"),
    ]:
        shutil.copyfile(path, root / name)
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                source_sha256=sha(source),
                hls_manifest_sha256=sha(hls / "manifest.json"),
                files={
                    str(p.relative_to(root)): sha(p)
                    for p in root.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts
                },
                scope="Original Decode xq_rope with explicit odd offset restored after base reset, plus matched product observations. Unrepaired source evidence230412 is intentionally retained. Wrapper copies immutable input to output before original in-place transform, within timed interval; HLS writes directly. No full Decode or hardware comparison.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", type=Path)
    p.add_argument("--execute", type=Path)
    p.add_argument("--worker", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.prepare.resolve())
    elif a.execute:
        execute(a.execute.resolve(), 1200)
    elif a.worker:
        root = a.worker.resolve()
        verify(root)
        sys.path.insert(0, str(root / "implementation"))
        from mesh_pair_rotation_sdk import run

        run(root)
