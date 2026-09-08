# Example tour

Read a frontend source together with its `PORT.json` and contract. Profiles are grouped by source provenance, while compiler/runtime code is shared.

| Topic | Where to start |
| --- | --- |
| Basic matrix product | [GEMM](../benchmarks/linear_algebra/sdk_examples/gemm/hls.cpp) |
| Distributed dense linear algebra | [GEMV contract](../docs/contracts/MESH-GEMV.md), [SUMMA](../docs/contracts/SUMMA.md), [Cannon](../docs/contracts/CANNON-CONTRACT.md) |
| Resident iterative solver | [512-variable CG frontend](../benchmarks/linear_algebra/sdk_examples/mesh_cg_512_4x4/hls.cpp), [contract](../docs/contracts/CG-CONTRACT.md) |
| Distributed transform | [FFT contract and profiles](../docs/contracts/DISTRIBUTED-FFT.md) |
| Supplied-Q/K/V attention | [HLS source](../benchmarks/inference/waferllm/attention_64x128_8x8/hls.cpp), [contract](../docs/contracts/RESIDENT-ATTENTION.md) |
| Explicit mixed precision | [Blocked MLP source](../benchmarks/inference/waferllm/mlp_128x128x512_8x8_blocked/hls.cpp), [contract](../docs/contracts/RECTANGULAR-MLP.md) |
| Resident multi-stage composition | [Projection/residual/RMS source](../benchmarks/inference/waferllm/projection_residual_rms_64x64_8x8/hls.cpp), [contract](../docs/contracts/PROJECTION-RESIDUAL-RMS.md) |

`generated/` includes actual emitted CSL from the three inference examples above. Each subdirectory contains `PROVENANCE.json` naming its qualified run. These are source-reading examples, not complete executable frozen bundles; use the runner to generate all metadata, bindings and dependencies for a fresh execution.

The math/state witnesses used by tests are under `tests/fixtures/`, with original-path hashes recorded separately. They are not invented expected outputs.

## New source-reading entries

- [Qualified 25-node QKV/pairs/cache-attention graph](../benchmarks/inference/waferllm/projected_cache_attention_3x256x512_8x8/hls.cpp): read-only supplied old cache, separate new K/V outputs.
- [35-node attention-plus-FFN candidate](../benchmarks/inference/waferllm/projected_cache_ffn_3x256x512x512_16x16/hls.cpp): implemented and undergoing full SDK qualification; not an admitted profile.
- [Mean-statistic RMS contract](../docs/contracts/MEAN-STATISTIC-RMS.md): explicit overflow handling, range proof and actual primitive SDK experiments.
