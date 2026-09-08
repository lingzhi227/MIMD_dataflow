# Next resident boundary: normalized feed-forward block and final residual

The first source-faithful boundary is WaferLLM Prefill `rmsnorm_z → z1_matmul → z2_matmul → z3_comp → h2_matmul → add_result`, pinned to `fd1c2daae37cd68706c03fc8009887ecee9900f8`, Apache-2.0. This finite13-node boundary now has three qualified profiles; the larger17-node tail now has a shared lowering and three qualified SDK profiles. See [PREFILL-OUTPUT-TAIL.md](PREFILL-OUTPUT-TAIL.md) for its frontend, middleware, resource and evidence contracts.

For supplied `Z[M,N]`, feature gamma, up/gate weights `[N,F]` and down weights `[F,N]`, the result is:

```
X = RMSNorm(Z, gamma)
U = X * up_weight
G = X * gate_weight
H = U * SiLU(G)
Y = Z + H * down_weight
```

`add_result` updates **Z**, the post-attention projection residual. It does not add the original pre-projection residual input, and there is no normalization after that final addition in the pinned source. The first lowering should express these ordinary typed operations as a 13-node graph, retaining explicit source two-hop placement and a declared down-accumulation policy. Source half versus explicit block-f32 down accumulation must remain different numerical policies.

Connecting the preceding qualified projection/residual/RMS gives the 17-node tail:

```
Z = attention_output * output_weight + residual
Y = Z + MLP(RMSNorm(Z, gamma))
```

This still starts with supplied attention output. It is not full Prefill, full Decode, a multihead/masked attention contract, or a KV-cache implementation.

## Storage and completion obligations

- Keep logical Z alive until the final residual addition. The current isolated postnorm output is only the normalized value; discarding Z is invalid for the larger graph.
- Normalized X is shared by the up and gate projections. Preserve the completed left-buffer ownership across the gate branch, as in the executed MLP repair.
- Join both communication axes and local compute before reusing projection DSRs, receive buffers or local normalization scratch.
- Reuse normalized-X storage for the down-projection result only after both up/gate consumers have finished. Z remains a distinct live allocation.
- Size shared weight/receive storage for the largest participating tile, not the first projection's tile. Diagnostic observations have a separate explicit budget.
- Preserve the original row collective and SDK math behavior, while checking the composed original-input mathematics separately. A narrow numerical range proof is not a relative-accuracy proof.

`region_lifetimes.py` now validates declared storage intervals and resource acquire/release events. The current projection/add/RMS plan maps its physical arrays to 13 logical lifetimes across six completion phases. This is a foundation for the larger graph, not proof that its missing lowering already exists. Compiler-managed temporaries, dynamic stack and arbitrary CSL control flow are outside that pass.

## Acceptance before registration

Require executable HLS C++, all frozen intermediate stages, a real generated-CSL SDK run, independent original-input mathematics, exact scheduled trajectories, source-control provenance, changed warm inputs and zero/cancellation resets, immutable public inputs, corruption rejection, matched local timing scope and linked memory. First qualify the 13-node normalized feed-forward boundary, then connect the 17-node projection tail while preserving Z. Keep full QKV/RoPE/attention connection, masks/heads, cache/state and full Prefill/Decode explicitly open in the backlog.

## Source execution preparation

`feed-forward-source-20260907T080836373657Z` freezes a three-call source control with repaired local RMS/row collective, original up/gate/SiLU flow with the gate ownership repair, explicit blocked-f32 down accumulation and original final `add_result`. Inputs are supplied Z, feature gamma and independently changed small half weights. The normalized tensor, up/activated-gate/down/final output and f32 accumulation are observed. `feed-forward64-source-preflight.json` passes model-vs-original-input checks before SDK execution. MLP delta is checked separately from the final residual output so a large residual cannot hide arithmetic failure. This source execution completed three SDK calls and passed both target bits and independent delta/final math. It adds no HLS catalog entry.

## 13-node implementation and precision experiment (2026-09-07)

The frontend and shared lowering now implement the 13-node graph in `projects/waferllm/feed_forward_64x64x256_8x8_all_blocked*/hls.cpp`. Each projection declares `accumulation=block_f32` and uses `matmul_blocked<f16,scalar>` with a block equal to the physical K tile. Half local DSR products remain; reusable `block_accumulate.csl` widens each completed partial, merges in f32, and narrows the cumulative result for the next consumer. The up/gate projections share an accumulator allocation with explicit resets; their final f32 snapshots remain separate for auditing. This is an explicit numerical policy, not silently interchangeable with the upstream half recurrence.

An earlier down-only policy failed the unchanged nonuniform-gamma cancellation fixture in the spatial target model: delta relative L2 0.56203, although the final residual passed. Actual C++ delta L2 was 0.00299455. Spatial half accumulation visits different K blocks in different PE columns, disturbing cancellation across those columns. Blocking all projections makes the target delta L2 0.00299455 on that fixture. All eight native and target delta/final preflight checks pass without changing data or tolerances. Real SDK completion is required before qualification.

Actual native intermediate observation uses the preserved Clang declaration byte range to insert one read-only output statement into a separate diagnostic executable. It uses frozen headers/runtime and original inputs, and must preserve original public outputs exactly. These files and stdout are part of the bundle manifest. Nested evidence paths are hash checked with traversal and escaping-symlink rejection. The first all-block counter attempt was rejected before SDK because the older manifest verifier accepted only top-level artifacts; it is preserved. A fresh bundle contains the fix.

`feed_forward_debug.py` exposes normalization (step0), shared projection stages (steps1–25 for P8), narrowed delta (26) and final residual (27). Counter mode marks absent half histories as unobserved; retained final f32 accumulators are actual observations, available at the last round of each projection. Completed-call checking is a partial diagnostic, never full qualification.

The general regression runner now applies actual native delta, predicted target delta, and final-output checks before SDK, and independently checks actual device delta after execution. Final residual accuracy alone cannot qualify this contract. Source control `feed-forward-source-20260907T090130004244Z` uses the same all-projection policy and completed three SDK calls, matching target half/f32 stages and original-input math. HLS counter090013451671 and sampled090228568221 completed eight calls each and are registered, along with larger128×64counter091416894851. See FEED-FORWARD.md for full accuracy, observation, mutation and performance scope. Counter static ELF high-water is20560B/PE (28592B remains statically unallocated), excluding dynamic stack.


The next 23-node supplied-Q/K/V resident boundary is now under SDK validation; see [RESIDENT-ATTENTION-TAIL.md](RESIDENT-ATTENTION-TAIL.md) for frontend, resource and debugger contracts. This does not reopen or invalidate the completed 13/17-node qualifications.
