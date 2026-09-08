# Triangular distributed Cholesky

The HLS profile `mesh_cholesky.v1` lowers a typed lower Cholesky factorization
to the official SDK's right-looking triangular wavefront. This is a specific
supported spatial algorithm, not an arbitrary C++ compiler or a substitute
for CSL. The source is SDK `benchmarks/cholesky` at commit
`4866cf330333446cb5e529e10f36be4600d1df29`; notices remain in all derived files.

```cpp
#include "spatial.hpp"
void design() {
  auto a = spatial::input<128,128>("a");
  #pragma csl dataflow rows=8 cols=8 triangle=lower update=right_looking fp=relaxed compute=vector
  auto result = spatial::cholesky(a);
  spatial::output("result",result);
}
```

The C++ header and independent f32 IR evaluator express a right-looking lower
factorization. Shape, symmetry and positive-pivot checks occur before SDK
execution. The dataflow annotation chooses a square PE mesh with square tiles;
only its lower triangle computes. An unsupported policy is rejected.

## CSL algorithm and lifetime

The original SDK algorithm remains recognizable: the diagonal pivot PE uses
`<math>.invsqrt_f32` and DSD scaling; a column multicast supplies left-fringe
PEs; row multicast then supplies trailing blocks. A paired row/column receive
joins through activation/unblocking before a DSD rank-1 update. Diagonal
blocks update only their lower triangle. Local tasks 17/18 and colors 0/1
retain their source roles. Queues 2/3 and 4/5 separate column and row traffic.

Retiring a block column sends two control wavelets to advance the row routing
switches. Consequently, resetting only the iteration counter would not make
the original one-shot code reusable. The generated host binding performs a
blocking all-PE `prepare` launch, then a separate `f_chol` launch. Preparation
resets iteration, DSD position, diagnostics and upper output storage, and uses
`<tile_config>.switch_config.clear_current_position` on switched row routes.
The barrier prevents new sends from reaching a neighbor before its reset.
The completed factor launch drains the previous invocation before preparation
is repeated. This mechanism reuses prior SDK 2.10.1 collective migration work
documented in the shared knowledge repository's
`reproduction/COLLECTIVES_FIXED_FAMILIES_SDK2101.md`.

The host distributes the input matrix; pivots, panel traffic and updates are
CSL work. Every returned matrix has its upper triangle zeroed on the device.
The factorization is no-pivot SPD Cholesky, with explicit relaxed FMA/invsqrt
arithmetic. No claim is made for singular, indefinite or ill-conditioned
inputs, and this profile does not perform an automatic diagonal adjustment.

## Validation and debugging

The four same-runtime inputs are dense SPD, diagonal-after-dense, a differently
scaled SPD matrix, and an SPD tridiagonal matrix. Validation checks the full
factor against independent float64 LAPACK with fixed `rtol=3e-5, atol=3e-6`,
exact zero upper triangle, positive diagonal and the full infinity-norm
relative `L*L.T-A` residual against fixed `3e-5`. No adaptive tolerance is used.

Two tile corners are exported after each active pivot's local update. A
separate global float64 right-looking computation checks these intermediate
witnesses. Unexecuted/inactive entries must remain zero. This is bounded
internal observation, not a full tile/panel trace. Progress counters also
check each PE's expected retirement pivot. `toolchain/debug.py BUNDLE --node
p2_3 --epoch 1 --step 12` selects one such witness and its retirement/timing.

`run-20260906T101310273331Z` passed the 32×32 / 4×4 PE profile for four calls,
including intermediate witnesses. Independent `math.fsum` reconstruction is
recorded in `coordination/cholesky32-review.json`. The earlier
`run-20260906T101034901607Z` retains a compiler type failure from passing a
scalar pointer to the timestamp array API; separate `[3]u16` arrays fix it.
The 128×128 / 8×8 PE run `run-20260906T101413537860Z` also passed four
calls, including 15,360 active corner observations. Independent reconstruction
is recorded in `coordination/cholesky128-review.json` (maximum relative
infinity residual 2.407597e-7).

## Performance evidence and remaining scope

Per-PE start/stop timestamps cover factorization and communication, excluding
host I/O and preparation. `experiments/cholesky_sdk_baseline.py BUNDLE` creates
a fresh copy of the pinned original SDK source, adds timestamp-only exports,
runs the identical first input, checks its defined lower factor and compares
matching per-PE timing boundaries. It preserves source hashes, complete raw
outputs and instrumentation. This baseline is explicitly native SDK source,
not an additional HLS port. Comparisons are simulator observations, not
physical-system results or end-to-end latency including preparation.

Measured native baselines are `evidence/native-cholesky-20260906T101823963828Z`
(32×32) and `evidence/native-cholesky-20260906T101840134537Z` (128×128).
The maximum active-PE factor interval is 15,264 vs 14,784 cycles for 32×32
(HLS/native 1.03247) and 176,776 vs 174,863 for 128×128 (1.01094).
These are maximum **local** start-to-retirement intervals, not a synchronized
host launch-to-completion timer. The full per-PE arrays remain in comparison.json;
some individual PE overheads are larger. Both baseline lower factors pass the
unchanged factor/residual checks. No end-to-end overhead bound is claimed.

The current HLS runtime always exports bounded corner history. An optional
production diagnostics mode, scalar comparison, larger/uneven meshes,
factor reuse/triangular solve composition and general graph integration remain
unimplemented. Measured baseline results must be consulted before making a
high-performance claim.
