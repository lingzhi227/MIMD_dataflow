# Example tour

Read a frontend source together with its `PORT.json` and contract. Profiles are grouped by source provenance, while compiler/runtime code is shared.

| Topic | Where to start |
| --- | --- |
| Basic matrix product | [GEMM](../ports/projects/sdk_examples/gemm/hls.cpp) |
| Distributed dense linear algebra | [GEMV contract](../ports/docs/MESH-GEMV.md), [SUMMA](../ports/docs/SUMMA.md), [Cannon](../ports/docs/CANNON-CONTRACT.md) |
| Resident iterative solver | [512-variable CG frontend](../ports/projects/sdk_examples/mesh_cg_512_4x4/hls.cpp), [contract](../ports/docs/CG-CONTRACT.md) |
| Distributed transform | [FFT contract and profiles](../ports/docs/DISTRIBUTED-FFT.md) |
| Supplied-Q/K/V attention | [HLS source](../ports/projects/waferllm/attention_64x128_8x8/hls.cpp), [contract](../ports/docs/RESIDENT-ATTENTION.md) |
| Explicit mixed precision | [Blocked MLP source](../ports/projects/waferllm/mlp_128x128x512_8x8_blocked/hls.cpp), [contract](../ports/docs/RECTANGULAR-MLP.md) |
| Resident multi-stage composition | [Projection/residual/RMS source](../ports/projects/waferllm/projection_residual_rms_64x64_8x8/hls.cpp), [contract](../ports/docs/PROJECTION-RESIDUAL-RMS.md) |

`generated/` includes actual emitted CSL from the three inference examples above. Each subdirectory contains `PROVENANCE.json` naming its qualified run. These are source-reading examples, not complete executable frozen bundles; use the runner to generate all metadata, bindings and dependencies for a fresh execution.

The math/state witnesses used by tests are under `ports/tests/fixtures/`, with original-path hashes recorded separately. They are not invented expected outputs.
