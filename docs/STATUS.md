# Release status — 2026-09-08

The captured index contains **141 bounded SDK-qualified profiles**, up from 122 in the initial curated release. This is a fixed source snapshot, not a live dashboard. Newer implementation code and unqualified candidates are included with their status explicitly stated.

## New qualified work since the initial release

| Boundary | Added qualified profiles | Validation scope |
| --- | ---: | --- |
| Projection/residual/RMS extensions | 2 | Counter and larger rectangular variants |
| Normalized resident FFN | 3 | Eight-call SDK runs, separate delta/final mathematical checks |
| Output projection plus FFN tail | 3 | Resident 17-node chain, source controls and resource checks |
| Supplied-Q/K/V attention plus tail | 3 | 23-node chain, single-head and unmasked |
| Mixed input-attention chain | 1 | 31-node bounded graph; explicit mixed precision |
| Batched RMS, normalized QKV, UP/GATE | 3 | Decode-layout numerical subgraphs |
| Batched complete FFN | 1 | Eight SDK/control calls, seven-stage accuracy and 52 fault rejections |
| Supplied-cache attention/output/residual | 1 | Eight SDK/control calls; read-only shared cache; 67 fault rejections |
| Batch-major adjacent-pair transform | 1 | Six SDK/repaired-source calls; original odd-offset failure preserved |
| Normalized QKV/pairs/cache-attention composition | 1 | 25-node graph; eight SDK/control calls, 11 numerical stages and 107 fault rejections |

The 25-node graph is `projected_cache_attention_3x256x512_8x8`. Its newly computed K/V are outputs; it does not append them to the old cache. All entries and original report links are in [the profile index](../validation/STATUS.md).

## Current implementation, not yet qualified

`projected_cache_ffn_3x256x512x512_16x16` connects the qualified attention boundary to normalized FFN in a 35-node resident graph. It adds an explicit mean-statistic RMS boundary, caller-owned CSL region composition, shared SDK planes, a 23-phase storage plan and a typed 50-port host/debug ABI.

Eight native inputs and 18 observed numerical stages passed on two hosts. The standard generated CSL and source-compute control compiled, with maximum linked static allocations of 42,064 and 43,984 bytes per PE; these do not measure dynamic stack highwater. Full eight-call SDK/control executions were still running at the captured development checkpoint. The retained early one-call audit and mutation results are partial evidence, not qualification of this full graph. The admitted count remains 141.

## Problems found and retained

- Long half accumulation failed mathematical checks in larger MLP/attention cases. Explicit block sizes and f32 merge policies are recorded; unchanged fixtures remain independently gated.
- A source DSD base reset discarded an odd-element offset. The failure was reproduced with an SDK probe; the source control uses an explicit repair.
- Legal attention output can overflow the subsequent half RMS sum. Actual SDK experiments show a prescaled-mean boundary remaining finite; this adds an explicit policy rather than silently changing old normalization semantics.
- Naive attention-plus-FFN storage exceeded the smaller mesh's per-PE budget. The new candidate uses a 16×16 layout, lifetime planning and actual ELF checks.
- Earlier one-call runs were deliberately stopped after measured diagnostic calls showed the old budget insufficient. Fresh full-eight runs use explicit six-hour budgets. A budget is not a completion guarantee.

## Remaining scope and user-directed stop

Full cache append/update, head/GQA selection, masks and automatic position semantics are not established by the current supplied-cache graphs. Nor are full-model weights, a production decoder or physical-wafer throughput delivered.

Finish and audit the accepted category-8 scope, report evidence and unsupported features, then stop. Categories 9–12, unrelated numerical backfill and Qwen require further user instruction. The legacy numbered queue preserves historical context; this stop condition supersedes automatic continuation.

Release CPU checks are recorded separately in [CHECKS.md](../release/CHECKS.md). Historical SDK success is not a claim that every profile was rerun with this release's current compiler.
