# Distributed dot and stable Euclidean norm

The `mesh_reduction.v1` frontend profile lowers contiguous distributed vectors
into SDK BLAS memory-DSD `@map` arithmetic and SDK `collectives_2d` communication.
It is a reusable numerical dependency for resident solvers, not a completed CG
application or a general BLAS implementation.

```cpp
auto x = spatial::input<131071,1>("x");
#pragma csl dataflow rows=8 cols=8 partition=contiguous reduce=row_column result=replicated fp=relaxed compute=map
auto norm = spatial::nrm2(x);
spatial::output("result", norm);
```

Dot takes two equal-shaped input vectors. Logical ownership is contiguous in
row-major PE order; each PE stores ceil(N/(rows*cols)) elements, and inactive tail
elements are zero. Local operations use the actual valid count. Scalar results
are replicated on every PE. The compiler checks shapes, ports, policy keys,
bounded local storage and the standalone resource contract before CSL emission.

Norm first computes a local absolute maximum, then a global maximum. A common
power-of-two scale is applied before squaring; the global scaled sum is square
rooted and rescaled. This avoids losing tiny vectors by unscaled f32 squaring.
SDK `reduce_fadds` provides SUM. MAX uses row/column gather and short root maxima,
because the installed collective API does not provide `reduce_fmaxs`.

The local BLAS source is copied unchanged, with its Apache notice, from
`projects/sdk_examples/upstream/benchmarks/conjugate-gradient/src/blas.csl`,
pinned SDK examples commit `4866cf330333446cb5e529e10f36be4600d1df29`.
Communication uses the installed SDK 2.10.1 library, not the older benchmark's
custom clock/allreduce protocol. Source notes are in `REDUCTIONS-SOURCE-NOTES.md`.

## Preserved execution evidence

| Profile | SDK run | Result |
| --- | --- | --- |
| dot8191, 4x4 PEs | 20260906T125948140744Z | Four calls pass |
| dot131071, 8x8 PEs | 20260906T130120779332Z | Four calls pass |
| norm8191, norm131071, norm17 | 20260906T130339029010Z | Four calls each pass |
| norm17, fourth input scaled by1e-30 | 20260906T131626747324Z | Four calls pass;47 empty PEs |

Each run retains frontend/checked/optimized IR, schedule, generated CSL,
implementation snapshots, native C++ outputs, original-vector independent
reference, SDK commands and results. Dot references use double `math.fsum` of
original f32 products; norm uses a double sum of squares and square root.
Independent coordinator review also used `math.hypot` for the original norms.
All PE result replicas, local partial witnesses, repeated-call progress and
local timing intervals are audited. Zero inputs must produce exact zero. Nonzero
norms additionally require relative accuracy3e-5 with **zero absolute tolerance**;
an underflowed zero cannot pass the ordinary absolute-error screen.

`native-reduction-20260906T130834040720Z` (dot) and
`native-reduction-20260906T131046502282Z` (norm) compare the same SDK primitive
adapter with diagnostic stores removed, on131071 elements/64 PEs/four calls.
All outputs match exactly. Maximum-local-interval HLS/native ratios are
1.0006475 for dot and1.0003612–1.0003621 for norm. These measure diagnostic-store
overhead of the **four-callback implementation in the recorded snapshots**.
They are not original full-CG performance, cross-PE elapsed time, host transfer
throughput, or CS-3 hardware measurements. Later callback implementations need
their own performance evidence.

## Current boundary and next composition

Supported meshes have2–8 PEs per axis, N1–262144, at most4096 local elements,
and1–16 calls. Inputs must be finite, exactly representable f32 values within the
configured integer absolute bound (at most32767). This is not a claim of full
BLAS floating-point range, arbitrary strides, complex arithmetic or root-only
results. PE memory accounting reserves8 KiB for SDK/control overhead and checks
a48 KiB envelope; it is an estimate, not measured executable memory usage.

The scalar wrapper now uses one callback state machine and accepts
explicit queue pairs. Solver composition still requires an explicit allocation
of local tasks, colors, queues, DSRs and microthread lifetimes. Queue parameters
alone do not prove safe coexistence with SpMV. Existing CSC SpMV produces row
ownership that differs from its input column ownership: resident recurrences
must redistribute vectors instead of reinterpreting their physical storage.

## Single-task callback validation

The updated wrapper binds one task, releasing three slots. Fresh SDK runs
131746578887 (norm17),131922409868 (both dots),132205677244 (norm131071), and
132337258035 (norm8191), all on2026-09-06, pass four calls with all existing
audits; norms now include1e-30 input. Phase is assigned before starting its
asynchronous SDK operation. Each completion advances once. The final callback
may immediately start the next reduction (norm MAX then SUM); no old-phase
writes occur after that call. This is single-flight state, not a reentrant API.
Alternative queue pairs are configurable but not yet validated in a combined
SpMV program. See RESIDENT-SOLVER-RESOURCES.md for the actual SDK task probes.

The single-task norm baseline `native-reduction-20260906T132946508931Z` now also
passes four calls on131071 elements/64PE, with exact HLS/native outputs and
maximum-local HLS/native ratios1.000359806–1.000360562 (about0.036% diagnostic
store overhead). It uses the updated1e-30 fixture and preserves the same
primitive-adapter scope; it is not a full solver or hardware comparison.

Resource-ledger correction during combined compilation: each SDK dimension's
two queue numbers are used in both input and output banks. The default complete
lists are input[2,4,3,5] and output[2,4,3,5]. Earlier schedules undercounted these
resources; the source emitted and executed all four in each bank. This does not
invalidate the standalone numerical results, but invalidates the earlier
proposed disjoint-queue composition. See RESIDENT-SOLVER-RESOURCES.md for the
preserved compiler rejection and SDK-based reconfiguration experiment.
