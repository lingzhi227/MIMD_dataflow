# Mean-statistic RMS over SDK collectives

The projected-cache attention graph is qualified as configuration 141. Its
attention result is the input to the next source-backed FFN boundary. This work
adds a **runtime primitive and range-certificate candidate**, not a new qualified
HLS graph. Existing HLS schedules retain their explicit sum-statistic policy.

## Actual failure mechanism and repair

For the legal coherent attention input recorded in
`evidence/projected-cache-ffn-next-boundary.json`, the attention result Z is 33.
With N=256 and eight feature partitions, each PE's 32-value half square sum is
34848. A global sum of 278784 cannot survive the final half conversion. Increasing
the collective accumulator precision alone does not repair that conversion.

`toolchain/runtime/sdk_axis_mean.csl` widens the local half statistic, scales it
by an explicitly checked power-of-two divisor in f32, calls the caller's existing
SDK SUM and broadcast providers, and only then converts to half. The existing
`batched_rms_local.csl` accepts `statistic_is_mean=true` to omit sum/N. Its default
is false, preserving the existing API and computation. This is an explicit
precision choice, not an algebraic rewrite enabled for all reductions.

The helper permits exact source/destination aliasing and borrows buffers until
its callback. It allocates two f32 arrays (8 × capacity bytes), local task 12,
zero colors, and zero queues. SUM, MAX and mean must be serialized by their
caller on the shared SDK planes. Callback completion releases local ownership;
it does not assert a global barrier. SDK/default memcpy reservations still apply.

## SDK 2.10.1 evidence

- `evidence/sdk-axis-mean-20260908T012837940552Z`: preserved compile failure.
  The queue-status observation needed a temporary before field access. No
  arithmetic ran in this attempt.
- `evidence/sdk-axis-mean-20260908T013530003341Z`: eight actual warm calls on
  8×8 PEs, both axes, extents 3/4, offset canaries, exact in-place buffers,
  divisor changes, skewed arrivals, SUM→mean→opposite-axis SUM handoffs.
- The first two calls observe ordinary half SUM becoming infinity while the
  new mean and normalized output remain finite. The other six also run the
  default RMS normalization path and compare its actual words.
- `evidence/axis-mean-independent-actual.json`: independent standard-library
  original-input math check of 49,152 actual half outputs. Maximum relative
  L2 error 0.00727670047, peak-normalized error 0.00932221041; fixed limits
  remain 0.02 and 0.03. Divisors 1/128/512 test generalized scaled statistics;
  only divisor 256 is the standard RMS expression for this geometry.
- `evidence/sdk-axis-mean-static-0135.json`: nine ELF classes; maximum linked
  static high-water mark 19,136 bytes. This includes the diagnostic harness and
  both RMS modes. It is neither marginal runtime cost nor dynamic-stack usage.

The timed repeat `evidence/sdk-axis-mean-20260908T014359000421Z` also
passed eight calls, with all prior raw ports unchanged.
`evidence/axis-mean-timing-review.json` records maximum per-PE mean latency
1,417 cycles for extent 3 and 1,453 for extent 4. These are instrumented
start-to-callback simulator observations including arrival waiting; they are
not an end-to-end benchmark or a speedup comparison against ordinary SUM.

## Range contract and limits

`mean_statistic_bounds.py` first proves finite local half products and sums.
For example, a 4-way decomposition with 64 local values of 33 is rejected: mean
scaling cannot repair an already overflowed local statistic.

Half-to-f32 conversion and a u16 power-of-two divisor (maximum 32768) are exact:
the smallest nonzero scaled half is 2^-39, still normal f32. The upper range uses
an outward enclosure of P-1 f32 additions and final half narrowing.

For standard RMS, the correlated candidate proof retains final-half mean
underflow explicitly. With T=sum(x²), N=P×Nt, u16=2^-11, u32=2^-24 and
eta16=2^-24:

    M >= (1-u16)^(Nt+2) (1-u32)^(P-1) T/N - 3 eta16/2

Enumeration of every permitted half M uses the pinned SDK inverse; Cauchy–Schwarz
bounds the normalized L1 while retaining gamma-first and final half rounding.
At |Z|≤34.5625, |gamma|≤1, N=256, P=8, epsilon=1e-6, the candidate gives
L1≤258.629557686 and element magnitude≤16.171875. This is a finite-range proof,
not a universal accuracy guarantee. Epsilon-sensitive accuracy needs actual
original-input tests, including values near underflow.

## Next HLS boundary

`evidence/axis-mean-ffn-storage-candidates-v2.json` explicitly retains observations
and allocates separate FFN buffers. The 8×8 candidate requires 68,248 bytes/PE.
The 16×16 candidate estimates 39,934 bytes/PE, including 4,096 additional bytes
reserved for FFN code/stack. This allowance is not measured sufficiency. The
P16 arithmetic range is an analysis domain and inherits no P8 SDK qualification.

Before enabling the composed HLS graph:

1. Carry the sum/mean statistic representation in typed lowering and reference
   execution; reject mismatched divisor and normalization contracts.
2. Prove all projections, SiLU/product and DOWN ranges and independently gate
   original-input errors. Preserve source-shared gamma and residual Z+delta.
3. Extend and verify the 16×16 schedule, MAX/SUM providers, DSR/task leases,
   transfer extents, observed storage and real linked memory.
4. Generate and run the whole HLS graph; compare a scoped source-backed control
   and measure device cycles. The primitive results do not establish complete
   Decode coverage, end-to-end FFN composition or real-hardware throughput.

## Subsequent 16-way and parent-graph checkpoint

Both new 16×16 SDK probes completed eight actual warm calls:
`evidence/sdk-axis-mean-20260908T015445740166Z` and
`evidence/sdk-axis-max16-20260908T015848730320Z`. Mean independently passes the
same 2%/3% original-input gate over 98,304 outputs (maximum L2 0.00833081674).
MAX checks both axes, all-negative remote winners including coordinates 8–15,
canaries and SDK plane handoffs. Their maximum linked static high-water marks
are 18,928 and 17,424 bytes respectively, for diagnostic programs only.

The frontend now recognizes an explicit `statistic=mean` schema. Its typed
boundary checks the actual gamma edge, shape, half type and input bound.
`projects/waferllm/projected_cache_ffn_3x256x512x512_16x16/hls.cpp` contains the
35-node composed algorithm. `projected_cache_ffn_ir.py` reuses the existing
25-node attention and 13-node FFN structural checks, including shared gamma and
Z residual. This source is an experimental candidate and has no PORT admission.

The parent planner derives Z≤34.0625 from its **own P16 attention range chain**,
then derives the mean normalization and FFN bounds: UP/GATE≤8.078125,
product≤65.25, DOWN≤130.5, final residual≤164.5. It reserves 39,934 bytes/PE and
checks 23 phases with 83 logical storage values. The frozen record is
`evidence/hls-projected-cache-ffn-parent-20260908T0212`. The complete graph's CSL
emitter, native/math checks, SDK execution, actual ELF/stack and source-relative
cycle comparison remain to be implemented and validated. 327 regression tests
passed before the final lifetime addition; affected tests cover that addition.

## Composed CSL implementation and ongoing full validation

The preceding 39,934-byte plan is preserved historical evidence. The first
linked whole-graph probe used 42,048 static bytes, exposing that reserve as too
small. The current planner reserves 44,030 bytes. With Q block1, the normal
compiler bundle `run-20260908T030001167477Z` links nine ELF classes with a maximum
42,064 static bytes; the pinned local-contraction control uses 43,984. These
figures do not measure dynamic stack use.

The complete graph now uses the public `mesh_projected_cache_ffn.v1` pipeline.
`projected_cache_region.csl` supplies a caller-owned completion callback and
delegated SUM continuation. `projected_cache_ffn_pe.csl` owns the outer launch,
clock, final residual and host completion. `sdk_axis_mean.csl` borrows the same
SDK planes at the RMS boundary; it adds a local completion task, not another
set of colors or queues. The mean consumer explicitly bypasses a second /N.

The Q block4 native cancellation case exceeded the unchanged 3% peak gate.
Q block1, retaining score block16 and the original inputs, passed. The eight
original and 18-stage observed C++ cases now pass on two hosts with exact public
output agreement. The standard runner's native observers and independent
stdlib preflight also pass in
`evidence/composed-standard-gates-20260908T032937748523Z`.

The full eight-call SDK run and source-compute control are still in progress.
The bounded earlier one-call experiments were intentionally stopped when their
one-hour budget proved insufficient for full instrumented I/O; their failed
records are retained. Fresh runs have a six-hour budget. Do not treat a
successful primitive, first call, preflight or native test as full composition
qualification. See the application README for current scope and evidence.
