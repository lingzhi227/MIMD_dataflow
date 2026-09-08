# Resident supplied-Q/K/V attention

This bounded dataflow computes `softmax(QKᵀ × scale) V` on one P×P region. Q, K and V are supplied half matrices M×N. It is unmasked, single-head attention. It does not implement tokenization, positional indexing, masking, multiple heads, KV-cache updates, full prefill/decode, or a complete model.

The C++ frontend uses existing transpose, matmul and softmax calls with three explicit dataflow policies. Transpose is a logical view. Score uses vertical two-hop K rotation and EAST-first horizontal feature reduction to rotating roots. Softmax uses the reusable synchronous `softmax_local.csl` library and SDK `@map` exp, with row max/sum collectives. The value contraction uses device vertical V alignment, horizontal probability alignment and a strided RHS DSD. No intermediate tensor travels through the SDK host.

## Typed composition and ownership

`mesh_attention.v1` reconstructs edges and canonicalizes independent input declarations. It verifies the score/softmax and device-matmul child contracts in the same region; synthetic fragment ports exist only during verification and CPU interpretation. Their names avoid collisions with user host names. They do not become SDK transfers.

| Resource/storage | Score | Normalization | Value product |
| --- | --- | --- | --- |
| Public Q/K/V inputs | Immutable; private K/V copies | Immutable | Immutable |
| Score S-half tile | Owned root output | In-place probability | Destructive horizontal left alignment |
| S-half temporary | Local partial, then cleared after reduction | Exponent workspace | Left receive scratch |
| L-half receive temporary | K receive scratch | Completed/inactive | V receive scratch |
| Right operand DSD | Contiguous, explicitly reset at each score invocation | Not used by local softmax module | Feature stride Mt, contraction increment one |
| Task19/20/25/26 and UT0..3 | Vertical join, UT2/3 active | Synchronous collectives after join | Both-axis joins and UT0..3 |
| Colors1..11 / queues3..7 | Shared source-derived route configuration | Reuse completed row channels | Both-axis traffic |

Detailed mode snapshots probability and exponents before destructive reuse, then records every value prefix and both current operands. Counter mode omits those copies and exports inactive zero placeholders; it must never be described as observing internal tensor values. Both modes check final output, immutable input handles, epoch/progress, roots, normalization completion and drained owned queues. DSR1 is reloaded for local compute; communication uses distinct source-derived DSR3/4 lifetimes.

The current planner checks shapes, packing, offset ranges and conservative PE SRAM budgets. Its descriptor and lifetime records are explicit schedule contracts, not a formal proof of arbitrary CSL programs. Actual runtime behavior is separately tested through repeated SDK execution and witnesses. General graph fusion and arbitrary aliasing are not implemented.

## Two failures retained as development evidence

The first source composition (`resident-attention-source-20260907T030854852839Z`) passed its initial call but failed the random second call:3584 score words and28669 partial words disagreed. Our V layout adapter left a strided right DSD; the subsequent score entry changed length but did not reset stride. Resetting the contiguous descriptor at every entry in fresh source031519726605 fixes all three calls. Source-exact intermediates and output pass; independent standard output relative L2 is approximately0.0011–0.0013. The zero-Q third case alone would have hidden the defect.

The first HLS counter bundle032017302871 executed six calls, but frozen audit failed because compiler packaging omitted the new runtime templates. Its independent output-only diagnostic passes; the bundle remains unqualified. The obsolete sampled and larger bundles were stopped once the shared packaging defect was identified. No failed evidence is overwritten. Fresh builds include both templates and run a frozen-only backend regeneration preflight at build and before SDK launch. Regression tests also remove an unlisted dependency or coherently rehash a changed target and require preflight rejection.

## Qualification status

Both bounded profiles are now qualified.64×128/P8 sampled032800549283 completes six SDK calls and78 coherent mutation rejections, observing1,661,952 internal half words. Qualification035517597578 includes actual C++/device independent original-input checks, frozen audit and matched source controls.128×256/P8 counter032941144043 completes six calls and39 mutations, with no internal tensor observation; qualification035216748909 records that scope explicitly.

Small sampled linked static maximum is22272B; large counter19392B. Remaining static space26880/29760B is not a runtime stack high-water measurement.

| Matched source control | HLS/source maximum local cycles | Scope |
| --- | --- | --- |
|64×128 counters|26938/27938|About3.58% lower; source scalar SDK exp versus HLS map, private K/V copies and completion counters|
|128×256 counters|89931/95370|About5.70% lower under this geometry; no internal tensor observation|
|64×128 sampled|28276/27873|About1.45% higher with unequal detailed observers; not a lean-performance comparison|

All three shared source inputs have identical final target-half output bits. Sampled score/probability/value prefixes and normalization scalars also match. The source includes explicit maximum repair, device V alignment/strided DSD and the warm score DSD reset. These are local WSE3 simulator intervals, not hardware throughput or complete model inference. The small sampled/counter HLS comparison is approximately4.97% additional local cycles for observation; earlier three-call partial reports remain separately identified.

The source origin is MeshInfra/WaferLLM commit `fd1c2daae37cd68706c03fc8009887ecee9900f8`, including its WSE3 adaptations, under Apache-2.0. Each generated bundle contains its license and a source notice. Experiments and numerical oracles remain outside compiler/codegen.


The corrected small counter032852989899 now completes all six SDK calls and frozen audit, with39 coherent mutation rejections. Matching sourcecounter032018473963 gives exact output bits for all three shared inputs; maximum local intervals are26938/27938cycles (about3.58% lower for HLS under this configuration). This counter control is linked by the subsequent sampled-profile registration. The source uses scalar SDK exp; HLS uses map, private K/V copies and additional completion counters. Counter results do not observe score/probability tensors.

`debug.py --check-completed` now supports attention's completed-call prefixes. It verifies the bundle with its frozen integrity checker, checks current schedule regeneration, records current diagnostic helper hashes and raw-result hash, and explicitly marks the output as not full qualification. A two-call prefix from actual SDK data passes; a request for full completion, missing launch, stale warm epoch and corrupted warm value are all rejected. Existing frozen runtime code remains unchanged.
