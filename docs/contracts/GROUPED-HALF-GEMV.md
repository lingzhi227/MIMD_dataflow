# Grouped half row-vector contraction

This is the source-distinct WaferLLM MeshGEMV schedule for `[1,M] @ [M,N]`, not the existing column-vector GEMV lowering. Pinned source commit is `fd1c2daae37cd68706c03fc8009887ecee9900f8`. The numerical compute, grouped two-stage reduction and broadcast come from `MeshGEMV/src`; original host packing is preserved under `references/waferllm-host/MeshGEMV` and root derivation under `references/waferllm-compile`.

## Expression and ownership

```cpp
auto a = spatial::input<1,512,spatial::f16>("a");
auto b = spatial::input<512,512,spatial::f16>("b");
#pragma csl dataflow rows=8 cols=8 broadcast=host_rows reduce=grouped_two_tree groups=4 result=replicated_columns fp=relaxed compute=dsr
auto result = spatial::matmul(a,b);
spatial::output("result",result);
```

Each PE row owns one X segment; the host replicates that segment across columns. W is distributed in row-major tiles. Local `@map` uses DSR1 fused half FMA over the segment. Group members reduce toward their midpoint, group roots reduce toward the global midpoint, then the result segment is broadcast to every PE in its column. Host gathering selects one row only after the auditor checks every replicated output.

| Mesh / groups | Group size | Group roots | Global root |
| --- | --- | --- | --- |
| 4 / 2 | 2 | 1, 3 | 3 |
| 8 / 2 | 4 | 2, 6 | 6 |
| 8 / 4 | 2 | 1, 3, 5, 7 | 5 |

Both reduction phases retain the source addition order: chains advance inward from both ends; a root consumes the lower side before the upper side. Changing group count can change rounding. The native C++ reference is increasing-K binary16 FMA; `fp=relaxed` explicitly permits the spatial order. All results additionally face the original-product half error bound and exact sparse-selection/cancellation witnesses.

## CSL resources and diagnostics

Queues 2/3 serve the first reduction, 4/5 the second, and 6/7 the broadcast, in both input/output banks. DSR1 belongs to local compute and source1 DSR2 to reduction. Source colors are retained. Logical PE coordinates are passed through layout parameters for stable warm initialization; exported input pointers remain separate from communication descriptor state.

The runtime records three phases: every local contraction, group-root results and the global-root result. Records at non-roots are explicitly inactive local-buffer snapshots, not partial reductions. The auditor compares only active phase records, all final replicas, per-phase/warm counts, queue status and local timestamps. `debug.py --node p0_1 --epoch 7 --step 2` identifies an inactive global-root record on the 4x4 configuration rather than labeling it a valid sum.

Planning rejects unsupported dimensions, non-divisible groups, groups with fewer than two members, a single group without the second phase, mixed precision and unsupported output ownership. Arbitrary roots, tails and composition with another concurrent lowering remain unsupported.

## Current qualification

All three configurations use eight inputs: signed random, exact row selection, zero reset, fused-vs-split rounding, minimum half subnormal, negative row selection, cross-group cancellation and distributed signed column selection.

| Configuration | Sampled SDK run | Active half records | Static bytes/PE |
| --- | --- | --- | --- |
| 128 / 4x4 / g2 | `182031000643` | 7,168 | 8,992 |
| 512 / 8x8 / g2 | `182658530022` | 45,056 | 15,344 |
| 512 / 8x8 / g4 | `182702944701` | 53,248 | 15,520 |

All are `run-20260906T<id>Z` in their corresponding project folder and pass eight SDK2.10.1 calls, active local/root bit audits, all output replicas and protocol checks. These static allocations are not stack high-water measurements. Separate earlier compile-only evidence remains preserved and is not relabeled as execution.

The 128 original-source control `evidence/grouped-native-20260906T183042389556Z` passes two changed-input calls with identical output bits. Its maximum-local interval is 512 cycles, versus 594 in sampled HLS (16.015625% overhead). Twelve adversarial corruptions are rejected; non-root inactive phase words are intentionally not claimed as reduction values. Independent source-order scalar reviews also agree on final bits for both 512 configurations.

## Optional low-overhead diagnostics

`--instrumentation counters` removes full phase snapshots and local-compute timestamp sampling. Per-phase/warm counts, queue state, every final replica and the total local interval remain. The audit explicitly reports `phase_records_observed=false`, `compute_timing_observed=false` and null active-phase/compute-timing results; the debugger labels missing observations.

128 counter run `183531848896` passes eight SDK calls. Its maximum-local interval is 521 cycles, 1.7578125% above the 512-cycle original-source control. Matched outputs and protocol data are identical to sampled mode; comparison is in `evidence/grouped128-instrumentation-comparison.json`. This is a simulator local-interval measurement, not hardware or synchronized global latency.

All three profiles are cataloged. Original-source512 controls `184727497942` (g2) and `185258636524` (g4) each pass two changed-input calls with identical output bits. Sampled HLS/source max-local ratios are1.0679405520 and1.0675958188.

512/g4 counter run `184909163273` passes eight calls: 1,445 maximum-local cycles versus1,532 sampled and1,435 original source, or0.6968641% counter-mode overhead. Static allocation is14,848/49,152 bytes. See `evidence/grouped512g4-instrumentation-comparison.json`. No internal-phase observations are claimed in counter mode. This closes these bounded schedules, not complete WaferLLM inference.
