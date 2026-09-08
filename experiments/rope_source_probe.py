"""Observe the exact pinned Prefill pair transform and its temporary DSD lengths."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_root = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / "hls-layout.json").is_file())
_bootstrap_sys.path.insert(0, str(_bootstrap_root / "tools"))
from bootstrap import configure, repository_root
configure(_bootstrap_root)


import argparse, datetime, json, shutil
from pathlib import Path
import numpy as np
from probe_runtime import read, sha, verify, execute, half_vector_worker

ROOT = repository_root(__file__)


def prepare(m, n, row_scratch=False):
    assert m > 0 and n > 0 and n % 2 == 0
    src = ROOT / "third_party/sources/waferllm/Prefill/src/prefill.csl"
    original = src.read_text()
    body = original[original.index("fn xq_rope()") : original.index("fn xk_rope()")]
    root = (
        ROOT
        / "validation/evidence"
        / (
            "rope-source-"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    )
    root.mkdir()
    length = m * n + n
    pe = f"""param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const seq_len_p_pe:i16={m};const _dim_p_pe:i16={n};const seq_len_p_pe_2:i16={2*m};
var data=@zeros([{length}]f16);var result=@zeros([{length}]f16);
var XQ=@zeros([{m*n}]f16);var ptr_XQ:[*]f16=&XQ;
var freqs_sin=@zeros([{n//2}]f16);var freqs_cos=@zeros([{n//2}]f16);var sin_val:f16=0.0;var cos_val:f16=0.0;
var dummy=@zeros([{m}]f16);
var seqLen_dsd_1=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{seq_len_p_pe}}->dummy[i]}});
var seqLen_dsd_2=@get_dsd(mem1d_dsd,.{{.tensor_access=|i|{{seq_len_p_pe}}->dummy[i]}});
"""
    for i in range(1, 5):
        pe += f"var X_tmp_{i}=@zeros([_dim_p_pe/2]f16);\nvar X_tmp_{i}_dsd=@get_dsd(mem1d_dsd,.{{.tensor_access=|j|{{_dim_p_pe/2}}->X_tmp_{i}[j]}});\n"
    if row_scratch:
        pe = pe.replace(
            "@zeros([_dim_p_pe/2]f16)", "@zeros([seq_len_p_pe]f16)"
        ).replace("|j|{_dim_p_pe/2}", "|j|{seq_len_p_pe}")
    pe += body + f"""
var progress=@zeros([1]u16);
fn prefill_struct() void {{}}
fn main() void {{
 for(@range(i16,{m*n})) |i| {{XQ[i]=data[i];}}
 for(@range(i16,{n//2})) |i| {{freqs_sin[i]=data[{m*n}+i];freqs_cos[i]=data[{m*n+n//2}+i];}}
 xq_rope();
 for(@range(i16,{m*n})) |i| {{result[i]=XQ[i];}}
 progress[0]+=1;sys.unblock_cmd_stream();
}}
var dp:[*]f16=&data;var rp:[*]f16=&result;var pp:[*]u16=&progress;
comptime {{@export_symbol(dp,"data");@export_symbol(rp,"result");@export_symbol(pp,"progress");@export_symbol(main);}}
"""
    (root / "pe.csl").write_text(pe)
    (root / "original-kernel.csl").write_text(body)
    (root / "layout.csl").write_text(
        """const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});
layout {@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});@export_name("data",[*]f16,true);@export_name("result",[*]f16,true);@export_name("progress",[*]u16,true);@export_name("main",fn()void);}
"""
    )
    rng = np.random.default_rng(210105)
    x = np.arange(m * n, dtype=float).reshape(m, n) / 16
    inputs = []
    for k in range(3):
        sine = (
            np.zeros(n // 2)
            if k == 0
            else np.ones(n // 2) if k == 1 else rng.uniform(-1, 1, n // 2)
        )
        cosine = (
            np.ones(n // 2)
            if k == 0
            else np.zeros(n // 2) if k == 1 else rng.uniform(-1, 1, n // 2)
        )
        v = x if k < 2 else rng.uniform(-2, 2, (m, n))
        inputs.append(
            np.concatenate([v.ravel(order="F"), sine, cosine])
            .astype(np.float16)
            .astype(float)
            .tolist()
        )
    (root / "inputs.json").write_text(json.dumps(inputs) + "\n")
    (root / "schema.json").write_text(
        json.dumps(
            dict(
                input="data",
                outputs=["result"],
                length=length,
                Mt=m,
                Nt=n,
                row_scratch=row_scratch,
            )
        )
        + "\n"
    )
    (root / "sdk-command.json").write_text(
        json.dumps(
            [
                "cslc",
                "layout.csl",
                "--arch=wse3",
                "--fabric-dims=8,3",
                "--fabric-offsets=4,1",
                "-o=out",
                "--memcpy",
                "--channels=1",
            ]
        )
        + "\n"
    )
    shutil.copyfile(Path(__file__), root / "driver.py")
    shutil.copyfile(ROOT / "experiments/probe_runtime.py", root / "probe_runtime.py")
    shutil.copyfile(ROOT / "lib/Runtime/sdk_process.py", root / "sdk_process.py")
    shutil.copyfile(
        ROOT / "runtime/csl/waferllm-LICENSE.txt", root / "WaferLLM-LICENSE.txt"
    )
    files = {p.name: sha(p) for p in root.iterdir() if p.is_file()}
    (root / "provenance.json").write_text(
        json.dumps(
            dict(
                source_commit="fd1c2daae37cd68706c03fc8009887ecee9900f8",
                source_sha256=sha(src),
                source_repository="https://github.com/MeshInfra/WaferLLM.git",
                files=files,
                scratch_length_repair=row_scratch,
                scope="Exact original xq_rope body and dimension-dependent temporary DSD lengths; copied host input and no-op outer continuation; no HLS application or standard RoPE correctness claim.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(root.relative_to(ROOT))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--row-scratch", action="store_true")
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--execute", type=Path)
    p.add_argument("--worker", type=Path)
    a = p.parse_args()
    if a.prepare:
        prepare(a.m, a.n, a.row_scratch)
    elif a.execute:
        execute(a.execute.resolve(), 600)
    elif a.worker:
        half_vector_worker(a.worker.resolve())
