# Resident projected-query cache attention: contract and verification

This bounded single-graph implementation passed eight actual SDK2.10.1 calls and is admitted by `evidence/qualification-20260908T010554710275Z.json`. It is the141st bounded configuration, not the141st complete application. The source is WaferLLM Decode commitfd1c2daae37cd68706c03fc8009887ecee9900f8, with the already documented RMS, softmax and pair-offset repairs. It composes normalized QKV, explicit adjacent-pair transforms and read-only-cache attention/output/residual. Its public outputs are the final residual, newly rotated K and newly projected V. The original source does not append these new K/V tensors into its supplied old cache. The graph must preserve that distinction.

The HLS source is `projects/waferllm/projected_cache_attention_3x256x512_8x8/hls.cpp`. There are25typed nodes:10inputs, RMS,3projections,2pair transforms, cache transpose view,3attention/output contractions, softmax, residual and3outputs. Names do not select lowering; `projected_cache_ir.py` validates this structural connectivity, shared normalization, shared coefficients and original residual. New K/V outputs prevent their computation from being silently discarded as dead graph branches.

## Mapping and runtime handoffs

Initial shape: B3,N256,S512 on8×8 PEs. X/gamma shard features onY and replicate onX. The three resident projection matrices shard input featureY/output featureX. After one fused SDK Y reduction of branch-major QKV, each PE holds all3batch rows of local featureX, replicated onY. Q/K pair calls use that storage directly. Q then contracts with cache K's feature-major block `[featureX,sequenceY]`; score reduces alongX. Softmax MAX/SUM run alongY, probability contracts with V and reduces alongY, then the output weight contracts alongX and restores featureY ownership for the original-X residual.

One imported SDK X/Y pair owns colors0/1 and4/5, queues2/4 and3/5, local tasks14/15 and16/17. SUM uses local10, MAX11. RMS/pair/local matmul enter only after the SDK callback clears ownership; pair borrows DSRbanks1–5 synchronously. Each callback means local provider completion, not an invented global barrier. No intermediate host transfer or route reconfiguration is required. Public old cache inputs and original X remain immutable throughout.

The callback chain has15phases: local RMS, Y SUM, normalization/local QKV, fused Y SUM, pairs/local score, X SUM, scale/MAX, Y MAX, exp/SUM, Y SUM, probability/PV, Y SUM, output projection, X SUM, residual/completion. Local RMS length4 transitions to packed QKV length288, score length192, MAX/SUM length4, then context/output length96. The lifetime plan must explicitly serialize shared workspace and DSR loans across these extents, and preserve the auxiliary new-K/new-V outputs through final readback.

## Derived domains and acceptance

Original input limits are X/gamma/cacheK/cacheV/cos/sin≤1, Q/K/V weights≤1/32, output weight≤1/8. A derived Q is not a supplied input with an assumed≤1bound. Use the executed RMS L1 certificate to bound each projection; propagate the explicit four-product pair transform before score. For softmax/PV, count the rounded probability-mass bound once across the full sequence, rather than multiplying its maximum by every sequence element. `positive_normalization_bounds.py` requires SDK half exp/reciprocal contracts already established by exhaustive probes; its integration still requires actual full-graph accuracy tests.

The contract experiment records exact-rational/upward bounds, the initial full-observation memory budget and source identity. These are finite-domain checks, not numerical correctness or performance evidence. Preserve fixed per-stage error gates, including normalization, Q/K/V, rotated Q/K, score, probability, context, output delta, final residual and auxiliary outputs. Keep the independent1%row-mass check. Observe actual C++ stages, actual half device products and all replicas; do not validate a large residual without validating its delta. Include zero/nonuniform gamma, identity/permuting coefficients, nontrivial angle, coherent and cancellation workloads, cache-axis sentinels and warm changed calls.

The required next steps are shared IR verification/planning, reusable runtime composition and codegen, actual native and target preflights, SDK execution/debugging, repaired source controls, fault rejection and linked memory/performance measurements. No full Decode, multihead/GQA, causal masking, automatic position generation or cache-update claim is implied.

## Implementation and current experiment (2026-09-08)

The candidate now lowers through the shared typed IR, schedule, range/lifetime
planner and CSL runtime. `run-20260907T235739454777Z` is the frozen Q4 candidate.
It has eight native C++ calls with eight separately compiled intermediate
observers, plus three public outputs, and eight target-model preflights. Each of
the eleven mathematical stages must meet relative L2 ≤2% and peak-normalized
error ≤3%, with probability row mass within1%. These are original-input gates;
intermediate rounding cannot be hidden by a large residual.

The earlier Q32 candidate `run-20260907T235016645218Z` failed its actual native
cancellation case: score L2≈4.419%. Changing only the Q projection block to4 gives
L2≈1.848% and peak≈2.868%; K/V remain32. Inputs, independent mathematical
reference and thresholds are byte-identical. The failure remains in place.
`evidence/projected-cache-native-q4-iteration.json` also records its historical
stage-only manifest mismatch; later failure sealing now has a regression test.
No failed snapshot was rewritten to manufacture a passing history.

`blocked_projection_bounds.py` propagates the correlated RMS L1 bound through
half blocks, f32 local accumulation, half narrowing and SDK reduction. It also
encloses the distinct native whole-row merge order. Q's certified reduced
magnitude is8.109375. The earlier contract report is an immutable Q32 design
snapshot, not the current schedule. The Q4 plan reserves46,816bytes per PE,
including shared collective workspaces, all sampled observations and16lifetime
phases (15compute/completion phases plus host readback).

SDK2.10.1 compiled nine ELF classes. The measured maximum static high-water mark
is46,592bytes per PE; the original-vecmat control is47,744bytes. These are linked
static measurements, not a dynamic stack bound. The SDK-host Clang17 rebuild
passed the same eight calls/eleven stages and regenerated all nine CSL files
identically. Real simulator execution completed eight calls, and all eleven original-input
mathematical stages passed on both native hosts and the device. All34 comparable
raw groups match the source local-compute control in every call;107 deliberately
corrupted snapshots are rejected. The qualification scope remains this shape,
domain, layout and precision policy.

The source control uses unchanged `gemv_static_step` and `vecmat_computation`
bodies from the pinned Decode source for all six contractions. Its zeroing and
DSD-base initialization are inside the measured interval. RMS, pair transforms,
softmax, f32 block merging, SDK collectives and instrumentation are shared. The
comparison measures local contraction lowering within this graph, not an
independent implementation of full Decode or real CS-3 throughput.

For inspection, `toolchain/projected_cache_debug.py` exposes25actual stored stages
on any PE and completed epoch. Missing observations stay unavailable. Partial
execution snapshots can be checked by the frozen incremental auditor, but only
process completion plus final raw audit, mathematical gates, ELF checks and source
control can admit a new library configuration.

`SOURCE-MAP.json` in the application folder now gives each source function's
exact line span and SHA256, checked against the pinned source and current HLS.
The final native-only guard build `run-20260908T003141050852Z` rejects duplicate
output names and noninteger certificate geometry. Its source, schedule, batches,
actual native stdout and nine CSL files match the executing235739 bundle;
`evidence/projected-cache-final-guards-identity.json` records the identity. This
new native build is not counted as another SDK execution.

## Next composition requires new numerical and storage contracts

A source-backed analysis witness is saved in
`evidence/projected-cache-ffn-next-boundary.json`. This is **not an additional SDK
case**. All original inputs at their declared coherent bounds predict a finite
attention result Z=33. Directly feeding this into the existing half-sum RMS for
the next FFN overflows the final half statistic: the exact sum is278,784. Thus a
successful attention graph and a separately successful FFN graph do not imply
that their declared numerical domains compose. A wider statistic or mean scaling
before final half narrowing needs its own proof and actual SDK experiment.

Likewise, naively appending only the FFN's resident weights adds12,288bytes perPE
to the present46,816byte sampled plan, reaching59,104before additional FFN code
or activations. This is a lower budget for naive concatenation, not a proof that
all possible compositions exceed memory. Larger meshes, explicit communicating
regions or justified storage reuse require new planning and SDK validation.
The source's final FFN residual uses Z, and both normalization functions load
W_dsd; neither a different residual nor independent gamma inputs may be silently
introduced as source semantics.


## Completed qualification

- Actual HLS: `run-20260907T235739454777Z`; source compute control:
  `evidence/projected-cache-source-compute-20260908T000650498000Z`.
- Eight calls on one runtime instance;5,748,736 numeric half words observed over
  all PEs. New rotated K and projected V are verified alongside the residual.
- Maximum original-input relative L2: score0.003121371, context0.009427662,
  output delta0.004134623, final residual0.000253392. Every stage satisfies the
  unchanged2%L2/3%peak gate; maximum row-mass error is0.006225586 (limit0.01).
- Every call: HLS65,606 versus source66,854 maximum-PE cycles; ratio0.981332456.
  The source wrapper's zeroing and DSD initialization are inside its interval.
  This is a matched local-contraction control, not full Decode or hardware throughput.
- All nine ELF classes are checked: HLS46,592/source47,744 static bytes perPE.
  The current HLS plan is46,816. Dynamic stack peak is not measured by this report.
- 107frozen mutation rejections,312unit tests, eight native stage observers on
  each of two hosts, and unchanged generated CSL after the final guard fixes.

The admission report binds full results, original source comparison, every
linked ELF, independent gates, source mapping, native-host provenance, failure
history and guard identity. Historical partial reports archive their inputs and
remain explicitly partial. The current native-only guard build is not counted
as another SDK run.
