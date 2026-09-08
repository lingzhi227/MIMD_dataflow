# Decode batch-major pair rotation

This extends the existing typed `rotate_pairs` operation with explicit batch-major feature ownership. It is a local numerical stage for later projection/attention composition, not an implementation of positions, head selection or cache updates. Source: MeshInfra/WaferLLM commit `fd1c2daae37cd68706c03fc8009887ecee9900f8`, `Decode/src/decode.csl`, `xq_rope` and the analogous offset `xk_rope`.

## Source semantics and layout

The source deliberately loads odd input into its first product. With `pair_order::odd_even`, the formulas are `even_out = half(half(odd*cos) - half(even*sin))` and `odd_out = half(half(even*cos) + half(odd*sin))`. Thus cosine1/sine0 swaps each pair. This convention is exposed in the HLS expression; it is not silently reinterpreted as the standard identity-at-zero-angle rotation. The existing `even_odd` convention remains distinct.

The new declaration is:

```cpp
#pragma csl dataflow rows=8 cols=8 partition=features axis=x layout=batch_major coefficients=feature_pairs compute=dsr fp=relaxed
 auto rotated=spatial::rotate_pairs<spatial::pair_order::odd_even>(x,cosine,sine);
```

For B5/N1024, PE(y,x) owns all five batch rows and features `[128*x,128*(x+1))`, in batch-major order, replicated along Y. Its 64 sine/cosine pairs are broadcast across the batch. Axis Y is also explicitly representable and has unit-level serialization tests; the current full application executes axis X. Unlike the older tiled Prefill profile, B need not be divisible by region height. Paired features must remain on the same PE. Per-token coefficients are not yet admitted for this new mode.

`mesh_pair_rotation.py` retains the existing profile, logical operator and native semantics, with an explicit layout branch. The older tiled default generates unchanged CSL. `mesh_pair_rotation_sdk.py` serializes feature shards, audits every replica and reconstructs logical output from actual readback. No host numerical operation participates in execution.

## CSL library and resource contract

`runtime/batched_pair_rotation_local.csl` exposes a synchronous `apply(input, output, cosine, sine, scratch, history)` operation. It processes a contiguous batch row with strided adjacent feature DSDs and four half DSR multiplications, then half subtraction/addition. Four rounded product vectors are complete before either result store. Scratch is four vectors of Nt/2 half values, independent of batch size. Sampled history is `[batch, product, feature_pair]` and records all four actual products.

The caller owns and exclusively lends destination/src0/src1 DSR banks1–5 for the whole call, after joining preceding SDK operations. No colors, queues, tasks or asynchronous operations are introduced. Input/output may be exactly aliased; scratch/history/coefficient arrays must be disjoint from both. Arbitrary partial input/output overlap is not supported. Descriptor and history offsets are checked against signed16-bit bounds. SDK memcpy still owns launch/I/O resources. The compiler's resource plan records these local leases; this primitive cannot run concurrently with a collective using the same banks.

The application wrapper preserves all three public inputs, writes separate output and exports complete products, result, warm-call progress and timestamps. Source-control wrapper copies the input into its output before invoking the original in-place function; this copy is deliberately inside its timed interval. The comparison therefore isolates the source arithmetic under matching observations while exposing the input-preservation difference.

## Qualified evidence

- HLS bundle: `projects/waferllm/batched_pair_rotation_5x1024_8x8_x/run-20260907T230157971748Z`.
- Original Decode control: `evidence/batched-pair-source-20260907T230412005447Z`. This original-source run completed but FAILED numerical equivalence. Its extracted function and complete provenance are retained. `evidence/batched-pair-source-offset-repaired-20260907T231236213768Z` is the separately executed repaired control.
- SDK-host native rebuild: `evidence/batched-pair-native-host-2306`, six independent component-mathematics checks and identical generated CSL.
- Static linked memory: HLS11,200 and source11,408 bytes/PE, three ELF classes each. Static allocation is not stack-usage measurement.
- Full unit suite:305tests passed in73.150seconds (including completed-prefix lifecycle rejection). Tests cover both feature axes word-by-word, replica ownership, rejected split-pair/policy cases, unchanged legacy CSL and explicit missing debugger observations.

Qualified as bounded configuration140 in `evidence/qualification-20260907T232229262682Z.json`. Six HLS and six repaired-source SDK calls passed, all six observed groups match word-for-word,491,520 actual half products were audited, and30 raw corruption tests passed. Maximum PE cycles are1,878 for HLS and2,081 for the repaired source control, ratio0.902450744834; the source input-preservation copy remains included. `evidence/batched-pair-full6-repaired-source-comparison.json` binds the complete comparison. This scoped simulator ratio is not hardware or model throughput. Inputs include identity/permutation coefficients, quarter-turn, nontrivial angles, zero, coherent cancellation and a changed final batch. Independent C++/device mathematics uses `.0015*(abs(product0)+abs(product1))+2^-23` per component; it does not hide cancellation behind a uniform relative-error threshold.

`evidence/batched-pair-alias-20260907T231538771192Z` passed eight actual SDK calls, covering30,784 output words across in-place/separate outputs, Q/K nonzero offsets, repeated descriptors and unchanged V/edge sentinels. Initial230755 failed because a scalar pointer requires explicit conversion to a many-item CSL pointer; that compiler evidence is preserved.

## Reproduction and inspection

`experiments/build_batched_pair_rotation.py` runs the shared HLS compiler and seals independent actual C++ output checks. `experiments/execute_frozen_bundle.py` executes its immutable snapshot in SDK2.10.1. `experiments/batched_pair_source_control.py` prepares/runs the original-function control. Comparison, fault testing and registration have separate drivers so a successful numerical run cannot silently bypass provenance or performance review.

The ordinary `toolchain/debug.py` pair-rotation stages0/1 inspect actual products/result. Batch-major plans report the whole batch range, local feature range and replica axis; an unobserved epoch fails explicitly. Further QKV/cache composition must derive propagated numeric ranges and join SDK collectives before this local DSR call. This standalone result does not validate that larger graph.

## Executed DSD correction, not a silent source substitution

The initial original-body control230412 completed six calls but failed the declared formula: resetting both DSD bases lost the initially declared odd offset1. Both descriptors then read/write even elements, while odd outputs retain the copied input. `evidence/decode-pair-original-offset-failure.json` records actual mismatches and independent mathematical failure. `evidence/dsd-base-offset-20260907T231325901420Z` isolates the cause in eight actual SDK calls: set-base reads even elements; explicit increment1 restores odd elements, retaining stride and extent.

The repaired control adds exactly `X_odd_dsd = @increment_dsd_offset(X_odd_dsd, 1, f16);` after its base reset. Its comparison driver requires that removing this one repair and product-observation block recovers the original function exactly. The HLS library constructs explicit per-row offsets and needed no numerical change. The original source's executed behavior must not be described as equivalent before this repair. Future K-slot composition must likewise restore the odd offset relative to the selected K-slot base.

## Incremental debugging improvement

Current pair transport atomically publishes each completed call, and the shared incremental checker validates complete prefixes, every replica and original-input mathematics. Partial success cannot qualify a full run; boolean runtime counts, forged early success and late replica corruption fail closed. `evidence/batched-pair-prefix-auditor-review.json` binds a current native-only rebuild with identical three CSL files and a full six-call reaudit of the original executed bundle. This is middleware validation, not another SDK qualification.
