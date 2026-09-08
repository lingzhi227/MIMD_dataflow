# Resident normalized feed-forward kernel

This source-derived numerical kernel enhances CSL through a typed C++ frontend, checked spatial schedule, shared CSL libraries and the Python SDK runtime. It is a finite WaferLLM Prefill boundary, not a complete model implementation or community endorsement. Upstream is MeshInfra/WaferLLM commit `fd1c2daae37cd68706c03fc8009887ecee9900f8`, Apache-2.0; the pinned `rmsnorm_z`, `z1_matmul`, `z2_matmul`, `z3_comp`, `h2_matmul` and `add_result` remain read-only references.

## Algorithm and physical mapping

`Y = Z + down(up(RMSNorm(Z,gamma)) * SiLU(gate(RMSNorm(Z,gamma))))`.

The 13-node frontend graph uses ordinary typed RMSNorm, matrix products, SiLU, multiply and add. Pragmas declare an 8×8 region, forward two-hop exchanges, double-buffered communication, local DSR compute and explicit accumulation policy. Weights are initially packed into the source-prescribed block ownership by the host. Normalization, all three products, gating and residual addition execute on the resident region without intermediate host transfers. The public Z must remain alive through the last addition.

For the qualified 64×64 → 256 → 64 counter profile, each PE owns an 8×8 input tile and an 8×32 hidden tile. Up/gate use K-block8 and down uses K-block32. Each local product uses half DSR arithmetic; a completed partial is widened and merged in f32 through `block_accumulate.csl`. The cumulative result is narrowed to half for the next stage. These are explicit numerical semantics, not the original half recurrence under a different name.

`rms_local.csl` retains SDK half math and explicit vector DSR operations. The frontend epsilon `1e-6` becomes binary16 bits17 (`1.0132789611816406e-6`) on target; the original-input mathematical check still uses the frontend literal and reports the resulting numerical error. `inference_comm.csl` and its routes retain the source row collective and two-hop transfer scheme. `gated_local.csl` implements the source local activation/product. Shared region hooks compose those libraries with the existing MLP engine rather than copying a second projection engine.

## Resources and completion

| Resource | Declared use |
|---|---|
| Application colors | 1–11 in the pinned route layout |
| Input/output queues | 3–7; SDK I/O remains SDK-managed |
| Local tasks | 19, 20, 25, 26 for joined projection progression |
| Explicit microthreads | 0–3 for paired memory/fabric transfers |
| DSR1 banks | Local product, normalization scale, gating and residual, in separate phases |
| DSR2 banks | Local square/sum; src1 for the subsequent row collective |
| DSR3/4 and 5/6 | Projection memory and fabric transfers |

`feed_forward_lifetimes.py` declares nine joined phases and 20 logical numerical lifetimes for the all-block profile. Public Z and weights survive the whole region. Private normalization storage is reused for the down result only after both consumers finish. Mutable rotating buffers and in-place transforms are described explicitly; this does not allow arbitrary SSA aliases. The 64-profile numerical storage inventory is7072B; retained observations, code and SDK allocations are counted separately. This pass checks declared lifetimes and explicit leases, not arbitrary CSL control flow, compiler/SDK temporary registers or dynamic stack.

The metadata extension adds the previously omitted normalization DSR2 phase. `evidence/feed-forward-lifetime-metadata-review.json` verifies all generated target files are unchanged for the three current FFN bundles; historical plans remain frozen.

## Current evidence

- Qualified counter: `projects/waferllm/feed_forward_64x64x256_8x8_all_blocked_counters/run-20260907T090013451671Z`; registration `evidence/qualification-20260907T091310420053Z.json`.
- Eight warm SDK2.10.1 calls pass exact normalized/delta/final half and three f32 accumulator checks, input immutability, task/queue progression, and separate original-input mathematical checks. Maximum delta relativeL2 .00550331; cancellation case6 .00299455. Final residual accuracy cannot substitute for the delta check.
-43 deliberate observation/protocol corruptions rejected. Linked static high-water20560B/PE; remaining static space28592B. Dynamic stack high-water has not been measured.
- Matched all-block source control `evidence/feed-forward-source-20260907T090130004244Z` completes three calls. Six corresponding half/f32 ports agree bitwise. HLS/source max-local cycle ratios are1.00879–1.00989. Input retention and observer/copy differences remain in these intervals.
- Separately, all-block source versus down-only source increases those first-three local intervals by8.06–9.10%, including added upper f32 observations. This is precision-plus-observation cost, not compiler overhead or evidence that the down-only policy passes all eight fixtures.
- Sampled64 `run-20260907T090228568221Z` now completes eight SDK calls,3,342,336 internal half observations and294,912 f32 values;64 corruptions rejected. Final half/f32 values agree bitwise with counter mode over all eight calls. Static high-water34064B/PE. Matched source max-local ratios1.03247–1.03633 include full prefix observations. Registration `evidence/qualification-20260907T094431763130Z.json`.
- Larger rectangular128×64→256→64 counter `run-20260907T091416894851Z` completes eight SDK calls,589,824 f32 observations and43 corruption rejections. Maximum delta relativeL2 .00528514; static high-water28720B/PE. Matched source `feed-forward-source-20260907T091432577918Z` ratios1.00582–1.00669 over three calls. Registration `evidence/qualification-20260907T094135951366Z.json`.
- The catalog now includes these three bounded FFN profiles.220unit tests and full124existing-profile native regression pass; the uniform regression entry has also exercised actual native delta gates for FFN. These counts are profiles, not distinct full applications.

All timing above is WSE3 simulator-local execution. It is not hardware throughput, end-to-end model performance or a general bound for arbitrary inputs.

## Reproduction and debugging

Use the catalog's `run_ports.py --select-exact waferllm/feed_forward_64x64x256_8x8_all_blocked_counters` for a fresh native regression; SDK execution requires the documented SDK host and flags. On the tested x86 host use `HLS_CLANGXX=/usr/bin/clang++-17` for the frontend/native stage; its default Clang14 does not support `_Float16` for that target. The variable selects one executable for both AST extraction and native compilation, independently of the pinned CSL SDK. `experiments/build_feed_forward.py --geometry M N F P --instrumentation counters` produces a fresh development bundle including actual native intermediate observation and target preflight. The source preparer is an external control, never part of compiler/code generation.

A bundle retains Clang AST/diagnostics, frontend/checked/optimized IR, semantic graph, schedule, generated CSL, frozen implementation, native command/stdout, separate native-observation artifacts, SDK command/options, actual results and audits. Do not rerun into a completed or failed bundle.

`feed_forward_debug.py` defines step0 normalization, steps1–25 MLP rounds/hidden stage, step26 narrowed delta and step27 final residual for P8. Counter half histories are explicitly unobserved; final upper/down f32 snapshots are retained actual observations. Prefer a bundle's frozen debugger when present. A completed-call partial audit is not full qualification. External diagnostic helpers and hashes are retained for older bundles whose frozen debugger predates this profile.

The next composition is the17-node supplied-attention-output projection/residual/FFN tail, preserving the post-projection Z for the final add. Full QKV/RoPE attention, masks/heads, cache/state, Prefill and Decode remain separate unfinished boundaries.
