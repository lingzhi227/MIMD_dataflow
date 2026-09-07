# Rectangular resident gated MLP

Two sampled ordinary-half profiles are qualified:64×64→256→64 on8×8 PEs and64×64→128→64 on4×4 PEs (`qualification-20260907T052033123403Z.json`). The catalog currently has119 entries with individually documented validation and algorithm scope; they are not119 complete applications. The larger128×128→512→128 ordinary-half profile failed the fixed mathematical accuracy contract and is not qualified. The numerical queue remains ahead of stencil/physics expansion.

The executable C++ frontend expresses four supplied half tensors and ordinary operators:

```cpp
auto x = spatial::input<64,64,spatial::f16>("x", 0.125);
auto u = spatial::input<64,256,spatial::f16>("up_weight", 0.125);
// Independent gate and down weights have the same explicit input preconditions.
#pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed
auto up = spatial::matmul(x, u);
```

The complete algorithm is `up=X U; gate=X G; hidden=up*SiLU(gate); output=hidden D`. `projects/waferllm/mlp_64x64x256_8x8/hls.cpp` contains all operations and policies. No MLP opcode or user variable-name dispatch is added. The graph verifier follows dependencies, canonicalizes independent inputs and accepts either multiplication operand order.

## Contracts and CSL lowering

`input_contracts.py` makes magnitude bounds executable in native C++ and validates frozen batches before execution. Its positive monotone half-FMA recurrence bounds signed contractions independent of block traversal order. Projection bounds, rather than just external inputs, must lie in the already validated local SDK SiLU domain (magnitude at most8). For the default input bound1/8, contraction64 gives gate/up bound1; contraction128 gives bound2. The hidden bound uses the target SiLU magnitude bound and rounded half multiplication. Down-projection overflow is checked separately. Broader numerical domains require additional evidence and an explicit implementation contract.

`projection_contract.py` shares rectangular shape, packing, descriptor and completion semantics across this composition and the existing normalized projection family. Static weights are packed into the source-prescribed initial block ownership; activations are logically tiled and aligned on device. Up/gate share a completed alignment with live pointer carry. Hidden receives a fresh alignment for the down projection. Each phase resets the right descriptor to contiguous features, then advances by its current output feature width.

The runtime reuses the pinned source communication/routes, with colors1..11 configured, queues3..7, local tasks19/20/25/26, microthreads0..3 and explicit DSR separation: compute1, memory endpoints3/4 and fabric endpoints5/6. See `INFERENCE-DSR-LEASES.md` for the bank-specific contract and correction of historical metadata. Compute and both communication axes join before the next buffer swap. Local gating executes synchronously after the gate join using the reusable `gated_local.csl` SDK `@map` SiLU and DSR1 multiply.

Public X/U/G/D arrays remain immutable. Private up becomes hidden at last use. Private X work becomes final output only after gate completes. Three weight inputs share one work/receive pair across phases. These aliases reduce SRAM without inserting intermediate host transfers. PE memory planning includes all immutable inputs, scratch, optional observations and code/control reserves; ELF static allocation must additionally be measured after compilation. Planned memory is not measured stack usage.

## Validation and inspection

Six native/SDK fixtures include changed dense weights, zero X, negative gates, zero down weights and simultaneous maximum positive operands. Target-order half witnesses are distinct from independent original-input `math.fsum`/`math.exp` arithmetic. Fixed numerical criteria are relativeL2≤.02 and peak-scaled error≤.03; these do not imply bitwise equality between different floating-point contraction orders. Sampled execution captures every projection prefix, gate/hidden/activated gate and first-round physical operands. Counter execution omits those tensors; the debugger must mark them unobserved.

Debugger steps0..P−1 select up prefixes; P..2P−1 select gate prefixes;2P selects hidden;2P+1..3P select down prefixes;3P+1 selects final output. Epochs are zero-based. Runtime operation diagnostics identify the current H2D/launch/D2H port. Completed-call checks can inspect a preserved prefix, but only a complete frozen SDK audit qualifies a bundle.

## Source provenance and present evidence

Reference: MeshInfra/WaferLLM commit `fd1c2daae37cd68706c03fc8009887ecee9900f8`, Apache-2.0, `Prefill/src/prefill.csl` functions z1/z2/z3/h2. Original source gate-branch left-pointer reset produces incorrect changed dense results. An explicit carry repair passes source probes64×64→256→64 (sampled) and128×128→512→128 (counter), each with three calls in one SDK runtime. Source reports `evidence/mlp64-sampled-source-review.json` and `mlp128-counter-source-review.json` distinguish observed stages from model-only references.

The generated runtime additionally preserves inputs and reuses storage; it must earn its own SDK correctness/performance qualification. Source-probe success is not that qualification. Full RMS/residual, prefill/decode models, head/cache semantics and hardware throughput remain outside this profile.

## Explicit precision policy and pre-execution gate

The larger ordinary-half run `mlp_128x128x512_8x8/run-20260907T044448791119Z` completed six SDK calls with exact target-order output bits. Its uniform maximum-positive case nevertheless had5.44% relativeL2 error against original-input mathematics, exceeding the unchanged2% criterion. Exact agreement with a target model is therefore insufficient. The failure is preserved in `evidence/mlp128-half-accuracy-failure-review.json`.

The opt-in down projection now has an executable typed expression:

```cpp
#pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=forward reduce=local overlap=double_buffer compute=dsr fp=relaxed accumulation=block_f32
auto output = spatial::matmul_blocked<spatial::f16,spatial::scalar>(hidden,d,64);
```

Each contiguous64-term block uses half fused accumulation. Block partials merge in float32, then the result rounds to half. The source-level native implementation traverses blocks in logical order; physical two-hop traversal can differ and is explicitly relaxed. The shared typed IR retains this arithmetic choice as matmul attributes. Currently only the MLP down projection lowers this policy, and its block length must equal the physical K tile. Unsupported combinations are rejected rather than silently changing precision or introducing a scalar fallback.

`block_accumulate.csl` uses SDK `dsd_ops` conversion and add, with typed float32 accumulation/scratch and a uint32 snapshot for SDK32-bit readout. It adds12 bytes per local output element. The primitive SDK probe `block-accumulate-20260907T054029601304Z` passes four repeated calls, including cancellation and reset. This primitive alone does not qualify the distributed MLP. Large counter and small sampled HLS/source controls are undergoing separate SDK validation.

Eight blocked fixtures retain the six original cases and add cross-block cancellation followed by a changed-input zero-down reset. Actual native stdout must pass independent original-input fsum/exp checks before SDK dispatch. The gate and reference are hashed into each fresh bundle. The unchanged ordinary-half128 native build `run-20260907T060649349984Z` is rejected at epoch5 before SDK. The blocked128 native candidate passes all eight; maximum relativeL2 is0.0038098126. No SDK or hardware conclusion follows from native success.

The modified source control shares the new precision library with HLS and includes the explicit gate-left carry repair. Its purpose is schedule/output/cycle comparison, not an independent arithmetic oracle. Independent native and original-input mathematical checks remain separate. Original long-half source controls cannot establish bit parity for this new arithmetic policy.

A physical-fabric mismatch was discovered in the first4×4 source comparison: the source used15×10 fabric while HLS used11×6. The old evidence remains preserved, but is not a claim of identical physical geometry. Fresh11×6 source control `mlp-source-20260907T055825818832Z` passes three calls with exact outputs and all sampled prefixes. Its maximum local intervals match the earlier measurement (109573 dense,91653 zero). `mlp64x64x128-sampled-source-comparison-v2.json` and `mlp-p4-fabric-correction.json` establish the corrected scope; the comparison tool now requires matching fabric dimensions and offsets.

The magnitude checks prove range safety under the stated half arithmetic model; they do not prove a uniform relative-error bound for every possible bounded input. In particular, cancellation can make relative error arbitrarily sensitive. The fixed2%/3% application criteria are tested acceptance conditions on the retained corpus, with separate target-order bit checks. Applications requiring stronger precision must select or add a suitable arithmetic policy and retain their own original-input tests.

## Completed blocked-accumulation qualification

Both fresh eight-call retries now qualify: `mlp_128x128x512_8x8_blocked/run-20260907T064549573557Z` and `mlp_64x64x256_8x8_blocked/run-20260907T064550728567Z`. Registrations `qualification-20260907T072500004338Z.json` and `qualification-20260907T073312186004Z.json` retain original-input native/device checks, frozen target audit, matched adapted source, corruption rejection and fresh ELF hashes. Large device maximum relativeL2 is0.0043642; fixed0.02L2/0.03peak limits were not relaxed. Static highwater is29408/27200 bytes perPE, excluding dynamic stack.

The counter profile omits half intermediate-prefix snapshots but retains final f32 accumulator readout (131072 values over eight large calls). The supplemental `mlp128-blocked-counter-observation-scope-correction.json` corrects historical comparison wording without changing its measurements. Source shares the precision library; original-input mathematics is a separate oracle. Matched max-local simulator interval overhead is approximately0.35–0.39% large counter and1.70–1.92% small sampled. These intervals include different observation/copy/schedule costs and do not establish hardware throughput.
