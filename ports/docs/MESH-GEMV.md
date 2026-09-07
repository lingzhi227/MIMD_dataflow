# HLS distributed GEMV

The first distributed linear-algebra port uses the SDK's `collectives_2d`
library and the algorithm in SDK `gemv-collectives_2d`. It computes f32 `y=A*x`;
the original example's bias term is outside this profile. This is not yet a
complete BLAS GEMV interface with arbitrary alpha/beta, strides or transpose.

## Frontend and dataflow

```cpp
#include "spatial.hpp"
void design() {
  auto a = spatial::input<256,256>("a");
  auto x = spatial::input<256,1>("x");
  #pragma csl dataflow rows=8 cols=8 broadcast=columns reduce=rows fp=relaxed compute=vector
  auto result = spatial::matmul(a, x);
  spatial::output("result", result);
}
```

The pragma is an experimental project syntax, not an industry standard. Its
semantics are explicit: partition A across a contiguous PE rectangle, scatter
x across the top row, broadcast vector partitions down columns, compute local
matrix-vector products, reduce partials across rows, and gather the results
down the rightmost column. The host distributes matrix blocks and supplies x
at the root; intermediate computation and collectives execute entirely in CSL.

`fp=relaxed` explicitly allows FMA and partitioned floating-point reduction.
The default unannotated matmul retains the previous implementation. This mesh
profile rejects unsupported shapes/policies instead of silently falling back.
`compute=scalar` provides a comparison implementation with identical communication.

Frontend AST and typed IR carry the dataflow annotation. `mesh_gemv.py` verifies
dependencies, divisibility and per-PE memory/DSD bounds, and schedules the six
stages. `mesh_gemv_sdk.py` compiles generated CSL and executes the SDK binding.
The shared build retains AST, checked/optimized IR, schedule, CSL, native C++,
inputs, references and implementation snapshots. This path uses the same
`run_ports.py`, manifest integrity and audit entrypoints as other profiles.

## CSL implementation

The runtime templates derive from official SDK examples commit
`4866cf330333446cb5e529e10f36be4600d1df29`; original Apache-2.0 notices remain in
the files. They import `<memcpy/memcpy>` and `<collectives_2d/pe>`, rather than
reimplementing those libraries. SDK collective defaults manage the x/y queue
and DSR allocations, as in the official GEMV example. The generated application
uses colors 0/1 and 4/5 and local task IDs 9–17; SDK memcpy has its own reservations.

Local computation walks matrix columns with a strided memory DSD and performs
`@fmacs` across the local rows. Unlike the original one-shot example, every call
clears the local accumulator. Only the rightmost column gathers row reductions;
other columns finish after their reduction completion callback. The original
upstream reference files are unchanged.

## Executed validation

SDK 2.10.1, WSE3 simulator, run `run-20260906T085103861321Z`:

| Matrix | PE mesh | Local tile | Compute | Epochs | Result |
| --- | --- | --- | --- | --- | --- |
| 64×64 | 4×4 | 16×16 | DSD/FMA | 4 | Passed |
| 64×64 | 4×4 | 16×16 | scalar control | 4 | Passed |
| 256×256 | 8×8 | 32×32 | DSD/FMA | 4 | Passed |

All inputs are independently generated f32 values. Validation compares native
C++ with source-order f32 IR, then checks final outputs against both that IR
and independent NumPy float64 results. Every PE's broadcast vector partition
is checked exactly, and every local product is checked against its independent
matrix block calculation. Repeated epochs exercise accumulator reset and
collective reentry. Floating tolerance is rtol=3e-5, atol=3e-6.

[SDK batch evidence](../evidence/run-20260906T085103861321Z.json) and
performance comparison (archive reference: `../evidence/mesh-gemv-cycles.json`) contain the details.
The complete 49-profile CPU regression passed in
`run-20260906T085409876123Z`; 25 unit tests passed. Earlier SDK validations for
other algorithms remain historical evidence, not a fresh full-SDK regression.

## Scoped performance evidence

For the same 64×64 matrix inputs, 4×4 mesh and communication implementation:

- Median measured local matvec: vector **765 cycles**, scalar **6861 cycles**,
  a **8.97×** ratio.
- Total simulator cycles: vector **90,003**, scalar **114,443**, a **1.27×** ratio.

Per-PE timestamp differences cover only the local matvec loop. Total cycles
include host transfers and diagnostic reads; both variants have the same
instrumentation. These are simulator comparisons with the scalar control,
not hardware throughput, not a tuned vendor-BLAS comparison, and not proof
of globally optimal mapping. FMA rounding differences are permitted by the
explicit numerical policy.

## Reproduce and inspect

```sh
.venv/bin/python run_ports.py --sdk --select mesh_gemv --sdk-timeout 600
python3 toolchain/debug.py projects/sdk_examples/mesh_gemv_256x256_8x8_vector/run-TIMESTAMP --node p3_4 --epoch 3
python3 compare_mesh_gemv.py VECTOR_RUN SCALAR_RUN -o evidence/mesh-gemv-cycles.json
```

`p<column>_<row>` selects a PE. Debug output includes its expected and observed
partial, vector partition and local timestamp words. The comparison tool
reaudits both frozen implementations before reporting performance.

Current bounds: mesh dimensions 2–8, evenly divisible matrix blocks, global
dimensions at most 512, per-PE memory budget and signed DSD offset limits.
Single-row/column meshes, uneven tails, persistent resident matrices across
host calls, release instrumentation modes and arbitrary graph composition
remain future work. Next in the linear-algebra queue is distributed GEMM/SUMMA,
followed by SpMV, reductions and solver composition; stencil expansion stays
behind the numerical-kernel queue.

> Packaging note: archive references identify original experimental artifacts or research references not bundled in this curated checkout. They are not local download links. See the root release selection policy.
