# Resident supplied-Q/K/V attention and numerical tail

This boundary combines the pinned WaferLLM Prefill score, softmax and value schedule with the qualified output projection and normalized feed-forward tail. It starts with supplied Q, K and V, and produces the final postprojection residual. The frontend expresses ordinary typed tensor operations and explicit dataflow policies; the backend composes shared CSL regions and libraries.

```
score = Q * transpose(K)
probability = softmax(scale * score)
A = probability * V
Z = A * output_weight + residual
X = RMSNorm(Z, gamma)
Y = Z + (X * up_weight) * SiLU(X * gate_weight) * down_weight
```

The last line means elementwise multiplication of the up and activated gate branches, followed by the down projection. All three MLP projections explicitly use half local partials and block-f32 accumulation. QK, PV and output projection retain the declared half recurrence. Neither precision policy claims full f32 products.

## Source and scope

Read-only reference: MeshInfra/WaferLLM, commit `fd1c2daae37cd68706c03fc8009887ecee9900f8`, Apache-2.0, `Prefill/src/prefill.csl` and its communication modules. Source controls document the existing RMS/row-collective and gate-ownership repairs, maximum-only initialization repair for stable softmax, and column-major V alignment. The descriptor stride changes apply only to PV; QK and subsequent projection stages restore contiguous contraction.

This is an unmasked single-head supplied-Q/K/V chain. Input RMS/QKV projections, RoPE, replicated heads, masks, cache state, full Prefill/Decode, and hardware throughput remain outside this boundary. A simulator observation does not establish CS3 hardware performance or community endorsement.

## Frontend and middleware

`experiments/build_attention_tail.py` authors the normal C++ HLS program. `toolchain/mesh_attention_tail.py` verifies the 23-node graph structurally, including the final addition to Z rather than the original residual, all nine input bindings, shapes, precision policies and a single consistent region. It builds child attention and tail plans; no numerical fixture is imported by the compiler.

QK uses vertical two-hop exchange and a rotating row-reduction root. Softmax uses the existing `softmax_local.csl` SDK-half math implementation and phase-joined row max/sum collectives. PV uses two-axis initial alignment and column-major contraction. The remaining projections use the existing shared DSR engine and block accumulator library. The runtime retains one task binding set, finite routes/colors/queues and one final host completion. Python performs input transport and observations, not intermediate numerical work.

`attention_tail_lifetimes.py` describes 16 joined phases. Q remains immutable in the physical x allocation; K and V have separate public allocations. Up/gate storage first serves logits/exponents and then probability/value receive buffers, before reuse by the MLP. The live Z allocation first receives attention output; a joined copy preserves A before output projection reuses that allocation. Score tiles must fit the borrowed storage. The pass checks declared lifetimes and resource leases, not arbitrary CSL control flow or compiler stack temporaries.

`attention_output_bounds.py` derives a conservative probability-mass and half-FMA output bound, including underflow error, from bounded V. The SDK exponential arithmetic model is exhaustively checked over finite nonpositive binary16 inputs, including exp(0)=1. The supported sum-size bound keeps the reciprocal normal. This range argument supports internal tensor contracts; independent original-input accuracy checks remain mandatory.

## Validation and inspection

Five separate C++ diagnostic executables observe score, probability, A, output projection and MLP delta, using frozen headers and Clang declaration ranges. Their original public outputs must stay bit-identical. Independent fsum/sqrt/exp checks evaluate all branches from the original nine inputs, including probability mass; final residual accuracy cannot hide an incorrect MLP delta. Predicted target arithmetic is checked before SDK execution. The real SDK auditor checks public input immutability, exact scheduled intermediate values, protocol counters, roots, queue state, f32 accumulators and completion lifecycle. A separate original-input device review runs before registration.

For P=8, `toolchain/debug.py --node p0_0 --epoch 0 --step N` selects:

| Steps | Actual observation |
|---|---|
| 0–7 | QK local partials, sampled only |
| 8 | Unscaled QK logits |
| 9 | Softmax probability |
| 10–17 | PV prefixes, sampled only |
| 18 | Resident attention output |
| 19–26 | Output projection prefixes, sampled only |
| 27–28 | Output projection and live Z |
| 29–56 | RMSNorm, shared MLP stages, delta and final residual |

`--check-completed` audits a single saved result snapshot with the bundle's frozen implementation. Before the first saved call it reports no available observations. Partial calls and unobserved counter histories never count as full qualification.

## Qualified counter profiles and sampled qualification

The first HLS profile `projects/waferllm/attention_tail_64x64x256_8x8_counters/run-20260907T114402035658Z` completed all eight SDK calls and is registered in `evidence/qualification-20260907T120153968936Z.json`. It passes five actual C++ branch observations, independent original-input device mathematics, exact scheduled half/f32 state and protocol audits, and87 corruption mutations. SDK-host Clang17 independently compiled and executed all eight native cases and five observations, generating byte-identical CSL.

Matched source control `evidence/attention-tail-source-20260907T112321109190Z` completed three calls. Eleven observed half/f32 port groups match exactly. HLS/source max-local cycles are116365/116531,116365/116534,107405/107574, or ratios0.998575/0.998550/0.998429. This establishes comparable local simulator intervals under matched inputs, precision, fabric and options; snapshot/copy differences remain included, and no hardware throughput is inferred.

Actual ELF maximum static high-water is26688B/PE, leaving22464B statically unallocated. This is not a dynamic stack bound. See `attention-tail64-counter-static-memory.json`, `attention-tail64-counter-source-comparison.json`, `attention-tail64-counter-original-input-math.json` and `attention-tail-audit-mutations-20260907T120110357065Z.json` under evidence.

Counter128/F256114821711402 is now also qualified after eight SDK calls and87 corruption rejections (`qualification-20260907T122757875519Z.json`). Its matched source115026419348 has eleven observed half/f32 port groups exactly equal. HLS/source local cycles193401/198135,193401/198134,175485/180211 give ratios.976107/.976112/.973775. Actual static ELF36720B/PE leaves12432B unallocated. Sampled64/F128115235234637 now also qualifies (`qualification-20260907T124511566933Z.json`) after eight SDK calls,135 corruption rejections,3592192 internal half observations and163840 f32 accumulator values. Its source115329737842 completed three SDK calls and original-input review. All11 observed port groups plus final up history match exactly; sampled/source ratios1.039085/1.038986/1.041291 include full-prefix observation overhead. Sampled static ELF37104B/PE leaves12048B unallocated. Full sampled64/F256 is rejected by the PE budget; the budget is not weakened to admit oversized instrumentation. All three profiles are registered.

All130 previously covered profiles regenerate byte-identical CSL under the shared backend (`evidence/attention-tail-prior130-codegen-review.json`). This check is code generation regression evidence, not130 new SDK executions. Complete regression suite243PASS plus four new input-prefix plan tests; the31-node CSL/runtime connection is still pending.
