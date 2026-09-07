# Stable distributed softmax on CSL

`spatial::softmax(x, scale)` computes a row probability distribution from scaled half scores. Its native C++ reference uses double-precision max-shifted exponentiation; the lowering explicitly selects half arithmetic and the SDK's half exponential.

```cpp
#pragma csl dataflow rows=8 cols=8 partition=tiles reduce=max_sum accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto probability=spatial::softmax(x,0.125);
```

The typed `mesh_softmax.v1` plan has six stages: scaled local maximum, maximum allreduce, max-shift plus SDK exponent/local sum, sum allreduce, reciprocal, and row-vector normalization. Every matrix tile is column-major, so each column is a contiguous row vector handled by DSD/DSR operations. SDK `<math>.exp_f16` remains the nonlinear computation primitive.

The generalized `runtime/row_chain.csl` performs either maximum or addition using the same five colors, input/output queues3/4/6, and source1 DSR2. Maximum and sum phases execute sequentially within one host launch. No host reduction or numerical repair is performed. The earlier qualified RMS backend retains its sum-only module; migration to this generalized module is a separate change requiring validation, not silently assumed from common source ancestry.

## Source behavior versus standard algorithm

Pinned WaferLLM `fd1c2daae37cd68706c03fc8009887ecee9900f8` Prefill initializes its maximum to zero. Actual source execution showed that uniform−1024 scores, scaled by1/sqrt64, yield all-zero exponents, zero sums, infinite inverses and4096NaN outputs. A following zero-input call recovers, ruling out persistent communication state as the cause of that case.

This HLS operation uses a finite lower bound−65504 for valid finite half scaled scores, then subtracts the true row maximum. It explicitly repairs the original initialization. Decode's fourth-power fast-exp routine is a separate approximation and is not used here.

The eight-PE exponential probe covers every31,744nonpositive finite half encoding, including negative zero, in two changed warm calls. SDK exp results, nearest-integer range reduction and half remainder match the source-derived model exactly.431exp outputs differ from nearest-half mathematical exp by1ULP. The combined production math model is rechecked against both exhaustive exp and sqrt observations; its domain and compiler settings remain explicit.

## Executed scope

-64×128 on8×8PEs, sampled bundle `run-20260906T221434624057Z`: six changed calls and64,512exact half intermediate observations. Random relativeL2 about0.0448–0.0474%, maximum row-mass error0.000933. Uniform negative, zero and row-constant inputs produce exact uniform probabilities. Sixteen audit mutations reject; linked static SRAM7376bytes, not runtime stack use.
-64×64 on8×8PEs, counter bundle `run-20260906T221829699155Z`: six changed calls pass. This matches the original Prefill score geometry and is used for the separately prepared corrected-source timing control.
-128×1024 is being prepared for actual execution. Full Prefill/Decode, masks, KV-cache composition and hardware throughput are not qualified by these standalone stages.

The auditor verifies raw input/outputs, target half bits, every sampled row statistic and exponent, phase/warm counters, owned queue drain and positive48-bit local timing intervals. Independent standard probability checks require relativeL2≤1%, peak-scaled max error≤1.5%, and row mass error≤1%. These are explicit half-precision criteria, not FP32 or per-component accuracy claims.

The debugger accepts a PE id, epoch and stage0–5, and reports logical row/feature ownership, raw probability or statistic bits, and exponent bits for stage2. Input, result and exponent storage have stable exported bindings across calls. Unsupported policies, scale/shape combinations and unsafe half difference bounds are rejected before code generation.

## Explicit elementwise mapping

The optional `elementwise=map` attribute lowers the SDK exponent stage to synchronous CSL `@map`, while preserving row communication and arithmetic. Omitting it retains the previously generated scalar-loop CSL byte for byte; the compatibility report covers all three scalar schedules. Compiler-managed descriptors are exclusive to this phase and are not offered for simultaneous sharing.

A matched8PE primitive experiment checks24,576output words exactly;512values perPE take36,382scalar versus25,105mapcycles. The complete64×64 counter softmax, bundle223559625301, checks six changed calls with identical outputs to scalar221829699155. Typical maximum local interval4990→4089cycles (18.06%reduction); the separated-peak case2943→1981. This is an application-level simulator comparison, distinct from the31%primitive interval reduction. The larger mapped case is now validated below.


The 64×128 sampled mapped bundle `223906593879` also passes six warm SDK calls. Every probability, row statistic, exponent and completion word matches scalar bundle `221434624057`. Typical maximum local interval is 9,450 → 7,394 cycles (21.76% reduction); the separated-peak case is 5,039 → 3,110. Both mapped and scalar 64×128, and scalar 128×1024, additionally pass the independent `math.exp` / `math.fsum` probability reference from actual C++ stdout and device results. Six fixtures include all-negative scores, zero scores, separated peaks, constant rows and changed random data. These fixtures now have a common regression entry, `distributed_softmax:M:N`.

The current semantic regression suite passes 122 tests. Scalar 128×1024 and mapped 64×128 each reject all 16 sampled-evidence mutations. The larger mapped execution and registration subsequently passed as recorded below.


## Larger mapped execution and catalog registration

Mapped 128×1024 on 8×8 PEs, bundle `224135809555`, passes all six SDK calls and 16 evidence-corruption tests. All probabilities, five row statistics, exponent matrices and progress words match scalar bundle `222220747022` exactly. Typical maximum local interval is 137,398 → 106,671 cycles (22.36% reduction); the separated-peak case is 68,038 → 37,312. Static ELF SRAM high water is 18,976 bytes versus scalar 18,992; this excludes runtime stack peak.

`qualification-20260906T231500546037Z.json` registers four profiles: scalar/map at 64×128 and 128×1024. Each frozen SDK auditor was rerun and actual native stdout plus device outputs were independently checked against standard `math.exp` / `math.fsum`. The catalog now contains 96 entries. `elementwise=map` performance claims remain scoped to matched WSE3 simulator local intervals; these measurements do not establish hardware throughput, global latency, positive-domain exponent accuracy or full attention coverage.
