# Distributed RMSNorm: algorithm, CSL policy, and evidence

Status: 64×128 on 8×8 PEs passes six generated-CSL SDK calls with 9,216 exact half stage observations. Three bounded sizes through128×2048/128PEs now pass and are registered. The full-domain SDK half-math experiment passes all 31,744 nonnegative finite encodings. The first compiler failure is preserved. No full Prefill/Decode or hardware-performance claim is made.

## User algorithm

`spatial::rmsnorm(x, weight, epsilon)` normalizes each matrix row:

`output[i,j] = x[i,j] * weight[j] / sqrt(mean_j(x[i,j]^2) + epsilon)`.

Inputs and outputs are binary16 tensors. Native C++ evaluates the mathematical operation with double intermediates and rounds the output to half. This is the independent high-level numerical meaning; it does not pretend that distributed half reduction has the same order or precision.

Example: `projects/waferllm/rmsnorm_64x128_8x8/hls.cpp`.

The explicit dataflow policy declares tiled ownership, feature-column weights, bidirectional row reduction, half accumulation, SDK half math, DSR computation, and relaxed floating-point equivalence. Unknown policies, unsupported shapes, nonpositive/underflowing epsilon, and unsupported precision fail verification. Wider accumulation is deliberately a future policy, not silently implemented as half.

## Typed middleware

`mesh_rms.v1` lowers to four inspectable stages:

1. Local row square sums, with a half rounding after square and after each add.
2. Allreduce across PE columns; both ends chain inward, root consumes east then west, one broadcast fans out.
3. Per-row mean, epsilon, SDK `<math>.sqrt_f16`, reciprocal.
4. Per-feature weight multiplication followed by per-row inverse scaling.

The frontend recognizes a typed operation, not a project or kernel name. `mesh_rms.py` validates the operation and describes this schedule; the runtime module takes a vector extent and reduction buffer. It is a candidate reusable communication primitive for other row statistics. Arbitrary multi-operation graph composition is not yet supported by this profile.

Local matrices are column-major so each local feature column is a contiguous row vector. Weights belong to the PE column and are replicated across PE rows. Row statistics are replicated across PE columns. The final output keeps the input ownership and stable exported storage.

## CSL implementation

- CSL DSDs perform full-tile half squares and vector row sums; DSR1 advances through feature columns.
- A standalone row reduction preserves the pinned Prefill chain's arithmetic and communication order. Color numbers are assigned explicitly (4–8), with alternating reduction colors and one broadcast color.
- The reduction owns input/output queues 3, 4, 6 and source-1 DSR2. Computation owns DSR1. No asynchronous user task is introduced. Synchronous fabric completion precedes host completion.
- Sampled mode exports all local sums, replicated global sums and inverse factors. Counter mode omits those records.
- Every call clears statistics and phase counts, retains a warm-call count, samples owned queue status, and records a 48-bit local interval. Python only transfers input, weights, diagnostics and output; numerical computation stays on the PE.
- The planner checks signed descriptor extents and reserves space for code/stack. Actual compiler static SRAM must also be inspected before qualification.

These resource assignments are exclusive within this region. They are not a general concurrent-region resource allocator.

## Source fidelity and corrections

Reference: WaferLLM commit `fd1c2daae37cd68706c03fc8009887ecee9900f8`, `Prefill/src/prefill.csl::rmsnorm_x` and `Prefill/src/comm_lib/comm_pe.csl::mv_allreduce_add_x`.

The original stage's final scaling uses `local_sum[i]` while iterating feature columns; standard RMSNorm requires the row factor vector. An executed source-only probe demonstrated this discrepancy. The original host also arranges weight segments by PE row; this implementation uses the correct feature-column ownership. Both corrections are explicit, rather than described as bit-identical reproduction of the original application.

A separate source control preserves the original full communication modules and square/sum/math code, changes the final scalar scale into a row-vector multiply, corrects host weight packing, and adds only a stage entry/completion/timing wrapper. Its `source-adapter.diff` is part of the immutable bundle. It is a **corrected source control**, not an unmodified upstream baseline.

See `WAFERLLM-INFERENCE-SOURCE-REVIEW.md` for the executed original-stage findings, including misleading receive-direction comments and the distinction between standard math and source fast-exp approximations.

## Validation contract

The native C++ output is compared against the high-level interpreter. Device qualification separately requires:

- Exact raw half bits for final target-order results and all sampled stages.
- Exact input/weight readback and logical-output packing, correct phase/warm counts, drained owned queues, bounded positive timestamps.
- Standard RMSNorm relative L2 error at most 1%, and maximum error at most 1.5% of the reference peak. These are preset bounded half-accumulation criteria, not FP32-accuracy claims.
- Non-square local tiles, varying signed feature weights, distinct row scales, changed random calls, zeros and small values that exercise half subnormals.

The target oracle models the actual SDK half sqrt source with non-contracted ordinary operators. The model passed an exhaustive SDK run across every nonnegative finite binary16 encoding (31,744 values, 62 warm calls). SDK sqrt differs from nearest half rounding for 6,375 inputs, by at most 1 ULP; the source-derived model matches every observed bit. Ordinary split expressions and explicit DSD FMA also match their separate models. This policy may need revisiting for another SDK release or compiler contraction settings.

The debugger selects `p<column>_<row>`, epoch and stage 0–3 and reports logical row/feature ownership and raw observed bits. Failed runs and analysis assumptions remain preserved; no failed bundle is edited into a passing one.

## Executed first configuration

`run-20260906T214109226421Z` passes all six cases. Ordinary-input relative L2 error is about 0.039–0.052%; the tiny-input case reaches 0.6116%, reflecting the explicit half epsilon/math policy. Zero input is exact. Thirteen diagnostic corruptions are rejected. Linked static SRAM is 6,544 bytes, excluding runtime stack use.

The corrected-source control `evidence/rms-corrected-source-20260906T214037481967Z` matches both raw packed outputs bit for bit. Sampled HLS / corrected-source maximum local intervals are 1513/1570 and 1518/1575. This is a bounded simulator comparison with different wrappers, not attribution to a particular optimization or a hardware speedup.

## Larger executed configurations

-128×1024/8×8PEs, sampled214324267081: six warm calls,18,432exact stage words, maximum local7394cycles, static18,192bytes.
-128×2048/8×16PEs, sampled214710421604: six warm calls,36,864exact stage words, maximum local7454cycles, static18,192bytes. The same five colors are used; more PE columns do not require additional colors.

Row-scaled cases reach about0.30%relativeL2; the tiny half-epsilon case remains about0.61%. Other changed random cases are about0.05–0.06%. Both configurations pass13corruption tests. These larger sizes have local timing evidence, not individual original-source overhead controls. Registration index222634742360 records native stdout and independent original-input math.fsum checks. Full92native regression222729791224passes.
