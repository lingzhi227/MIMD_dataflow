# Finite implementation queue

## Current authorized stopping point — 2026-09-08

The user explicitly instructed, in task `01a07763-54c2-74d0-87b3-f21cc4085999`,
to **stop after category 8**, report its verification evidence and unsupported
scope, and wait for new instructions before the following four categories.
Do not start categories 9–12, Qwen, or other applications. The proposed SDK
residual/norm backfill is a deferred candidate, not work to start after this
checkpoint. Earlier instructions to continue through the entire finite queue
are superseded by this stopping condition.

Category 8 closure requires a source-backed coverage audit of its numerical
stages and accepted Prefill/Decode compositions, plus the outstanding complete
SDK/native/mathematical/protocol/performance gates. The currently running
35-node profile is one obligation, not automatic completion of the category.
Missing heads, masks, cache update and full-model semantics must be explicitly
distinguished from what the pinned source actually implements; do not invent
new model requirements or label a bounded numerical graph a complete model.

Order follows the user's instruction. A completed bounded profile does not
close an entire source project. Read each source and its PORT.json before work;
update this queue as evidence is obtained, with source/schedule gaps explicit.

Every entry requires source-expressed HLS dataflow, typed intermediate stages,
generated CSL using SDK 2.10.1 facilities, native/independent numerical checks,
repeated execution, device intermediate-state audit and scoped performance
evidence. Preserve failures; a mathematical equation alone is insufficient.

| Order | Work item and source family | State / acceptance gap |
| --- | --- | --- |
| 1 | SDK distributed GEMV / collectives_2d | Bounded 64×64 and 256×256 complete with repeated SDK execution and vector/scalar comparison. General BLAS modifiers and tails remain uncovered. |
| 2 | SDK distributed GEMM/SUMMA / gemm-collectives_2d | Bounded 64³/16 PE vector+scalar and 128×256×128/64 PE vector complete: four warm SDK calls, prefix audit, scoped simulator comparison. Native large fixed-screen failure is explicit; general BLAS/tails remain uncovered. |
| 3 | Matrix-algorithms distributed LU, QR and SDK Cholesky | Cholesky and blocked no-pivot LU bounded32/128 complete: warm SDK, factor/reconstruction, sampled pivot witnesses, scoped native baseline. LU domain is strict row diagonal dominance. Blocked R-only QR32/square128/rectangular128×64 now passes four warm SDK calls and native comparisons; counters mode ~0.55% max-local overhead, sampled ~17%. Shared square-host regression passed; Q/pivoting remain uncovered. |
| 4 | SDK distributed sparse SpMV and collective dot/norm | Canonical CSC train SpMV512²/16PE and4096²/64PE pass four warm SDK calls with changed structure, empty partitions, integer index transport, full original-entry outputs and sampled partials. Aligned u16 metadata staging is probe-verified; source-native comparisons and targeted large repeat pass, with explicit floating-order scope. Distributed dot/norm up to131071 elements/64PE pass four SDK calls, all-PE witnesses and same-primitive baseline; norm17 also passes1e-30 input with47emptyPEs. Single-task collective FSM passes all five profiles; combined solver composition remains pending. |
| 5 | SDK CG, PCG, BiCGStab and power iteration | Existing small dense profiles insufficient. Combined transpose/SpMV/reduction probe passes16PE and restricted-capacity64PE; large has only320 static bytes below48KiB. Resident CSC CG512/16PE now passes eight SDK calls with independent original-system residual/history/protocol checks and0.83–0.97% diagnostic-store control overhead. Complete static footprint45376/49152; stack and original seven-point schedule/performance parity remain open. Jacobi-PCG512 now passes9warmSDKcases, weighted/unweighted residual audit and same-input diagonal control (4.1524 max-local ratio, scoped to that system). BiCGStab512 now passes10warmSDKcases, early/full/failed-stage audit and selected-store overhead control(~0.48–0.58%). Fixed-step power512 now passes8warmSDKcases,14mutation rejections and same-schedule scalar normalization control(~1.0091 scalar/vector max-local ratio). Completion is not eigenpair convergence; original seven-point schedule and larger shapes remain open. |
| 6 | Matrix Cannon and WaferLLM MeshGEMM/MeshGEMV schedules | Cannon64/4x4 vector+scalar and128/8x8 now pass four warm SDK calls, cyclic-prefix/block witnesses and native controls. Large warm-export defect exposed and fixed with invariant host handles; preserved failure. Vector/scalar localcompute~7.114× and native overhead~1.12–1.14% are scoped simulator results. WaferLLM two-hop f16 MeshGEMM64/128/256 now pass six warm SDK calls, exact scheduled half prefixes, mutations and pinned native controls (~8.9%/~3.2%/~3.25% sampled max-local overhead). Optional64 counter mode preserves final/ownership checks;256/8x8 full-prefix audit checks3,158,016 internal half observations. Source-distinct grouped half MeshGEMV128/512 (two group choices) passes eight warm SDK calls, local/active-root and final-replica audits, mutations and original-source controls. Counter-mode overhead128/512g4 is~1.76%/~0.70% max-local; sampled512 overhead~6.8%. Bounded schedule family closed; full inference remains item8. |
| 7 | Distributed FFT / SDK and available research sources | Bounded SDK C2C16/32/64 complete:16³ six direction/norm combinations,32³ sampled/counters and distinct five-stage transposed ownership,64³/256PE four warm calls. Exact source-device output controls at16/32/64;64 sampled overhead0.0302%max-local.16/32device roundtrips, original-domain directDFT, stage endpoints, protocol and mutation audits pass. Ten profiles registered205519409769. General planning, research-source identity and transposed-input composition remain explicit gaps; no hardware claim. |
| 8 | WaferLLM remaining numerical stages and prefill/decode composition | Bounded RMSNorm, softmax, normalized projection/fan-out, QKᵀ, score→softmax, device-aligned probability×V, supplied-Q/K/V unmasked attention, ordinary/blocked MLP and projection/residual/RMS profiles are qualified with per-profile source controls and numerical limits. The13-node normalized FFN64 counter now qualifies after8SDKcalls, separate actual native/device delta checks,43mutations and same-precision source comparison; sampled64 and larger128×64counter also qualify after eight SDK calls each. See the latest checkpoint and [FFN contract](contracts/FEED-FORWARD.md).17-node supplied-attention-output tail now has three qualified profiles. The23-node supplied-Q/K/V attention/output/FFN chain is under real SDK validation; see [resident attention contract](contracts/RESIDENT-ATTENTION-TAIL.md). Full QKV/RoPE connection, masks/heads/cache and full Prefill/Decode remain unfinished. |
| 9 | Stencil library lowering and wse-stencil | Parked until numerical queue. Existing resident stencil and native-library baseline are bounded. Need optional library call/FP/layout contracts and supported stencil families. |
| 10 | Physics simulation workflows from wse-stencil/SDK | Local apply kernels insufficient. Recover field staggering, boundaries and time evolution; validate conserved/diagnostic quantities where applicable. |
| 11 | Monte Carlo transport | Lookup only currently. Recover particle/control/RNG ownership and full supported trajectory workflow; statistical validation plus deterministic audit. |
| 12 | Simulated annealing and remaining application workflows | Flip/acceptance only currently. Recover distributed proposal, RNG, state updates and complete optimization trajectory. |

Sources are pinned in `evidence/source_inventory.json` (332 reference files,
not 332 verified kernels). SPADA and spatial-collectives communication patterns
are dependency work within the above entries, not an excuse to move stencil
or unrelated applications ahead of linear algebra. AMD HLS repositories remain
optional syntax/design references; CSL and SDK are the actual target.

Do not declare the project complete until the finite queue is audited against
the original inventories and remaining unsupported work is explicitly resolved
or reported as an external prerequisite. Maintain narrower per-profile claims.

## Latest item8 checkpoint — 2026-09-07

Blocked MLP64×64→256→64 and128×128→512→128 are now fully qualified after eight warm SDK calls, unchanged original-input accuracy gates, exact half/f32 observations, matched source controls, corruption rejection and linked ELF measurements. Pure-half large accuracy failure and earlier wall-time failures remain preserved. Counter instrumentation retains final f32 accumulator readout; it omits half intermediate prefixes.

Projection→residual→RMS now qualifies64sampled,64counter and128×256sampled after8SDKcalls each. Source descriptor and row-inverse defects are isolated; repaired source and extracted CSL library agree. Source comparison, original-input math, mutations and static memory are recorded per profile.

The13-node supplied-Z normalized FFN `Y=Z+MLP(RMSNorm(Z,gamma))`64counter is registered. Unchanged cancellation inputs exposed the difference between native and spatial half accumulation; explicit block-f32 up/gate/down solves that case without loosening tolerances. Eight actual SDK calls pass, including a separately observed delta; matched source local overhead~0.88–0.99%.220unit tests and full124old-profile native regression pass, plus targeted newFFN native regressions. Sampled64 and larger128×64counter now qualify after eight SDK calls each,64/43mutation rejections and matched source controls. The sampled profile observes3,342,336 internal half values; larger counter observes589,824 f32 accumulator values. See [FFN contract](contracts/FEED-FORWARD.md).

The17-node supplied-attention-output projection tail now has three qualified profiles (64counter,64sampled,128×64counter), eight real SDK calls each,54/82/54 corruption rejections, matched source controls and linked memory. Counter source overhead is~1.00–1.12% at64 and~.62–.71% at128×64 in local WSE3 simulator intervals. Full127prior-profile actual native regression passes with identical before/after CSL;227distinct unit tests pass (226full plus affected tail tests). All three SDK-host Clang17 native rebuilds and original-input projection/delta/final gates pass and reproduce executed CSL bytes. See [output tail implementation](contracts/PREFILL-OUTPUT-TAIL.md). Preserve post-projection Z through the final source `add_result`; do not substitute the original pre-projection residual or normalize after the last add. [Resident tail design](contracts/RESIDENT-PREFILL-TAIL.md) records this boundary. All three17-node variants are qualified; full attention connection, masks/heads/cache and full Prefill/Decode remain open. Complete this numerical queue before expanding stencil and physical simulation.

### Next concrete boundary after the 17-node tail: supplied-Q/K/V resident attention plus output tail

The23-node structural composition now has one qualified64/F256 counter profile (eight SDK calls,87 corruption rejections, three matched source calls, ELF26688B/PE, and actual SDK-host Clang17 checks). Rectangular128/F256 now also qualifies after eight SDK calls,87 mutations and matched-source ratios.974–.976. Sampled64/F128 now qualifies after8calls,135 mutations,3592192 internalhalf observations and three matched source calls (~3.9–4.1% sampled interval overhead). The31-node source control passes3SDKcalls; its HLS source proposal and unselected25-phase typed/lifetime plan exist, while CSL generation and native/device gates remain next. See [resident attention implementation](contracts/RESIDENT-ATTENTION-TAIL.md). This is not a full-attention/model claim. Pin the same WaferLLM commit and connect its `score_matmul → softmax_score → output_matmul → h1_matmul → z_add → rmsnorm_z → z1/z2/z3/h2 → add_result`. The public inputs are Q, K, V, output weight, original residual, feature gamma, and up/gate/down weights. The frontend must explicitly describe `score=Q*K^T`, scaled softmax, `A=probability*V`, then the qualified `Z=A*O+R; Y=Z+MLP(RMSNorm(Z,gamma))` boundary. Start with a declared single-head, unmasked, supplied-Q/K/V contract.

Required physical handoffs, using the existing executed resident attention and tail as controls:

1. Finish all score root reductions before score partial storage becomes the exponent buffer. Keep the key-transpose access contract explicit; complete row max and row sum collectives before probability is consumed.
2. Preserve the existing device-side vertical V alignment and horizontal probability preshift. Value contraction reads column-major V tiles through the right DSD with stride `Mt`, incrementing its contraction offset by one. Dead K receive storage and dead exponent storage can serve the value receives only after their producers complete.
3. After both value communication axes and local compute join, its `output_tile` is the logical column-major attention output tile `[Mt,Nt]`. Bind or copy this resident tile into the output-projection left working buffer; do not round-trip it through Python or reconstruct it from host inputs. The public output weight retains source-prescribed initial block ownership.
4. Reset output-projection `Mt=M/P`, `Kt=Nt=N/P`, contiguous right DSD access and projection output pointers. Value's strided descriptor must not leak into this ordinary product. Reset score entry descriptors again on the next whole invocation. Observe the value result and first output-projection operands to validate the transition.
5. Use one phase dispatcher and one binding of shared colors/queues/local tasks/microthreads. Attention's current terminal callback must continue into the tail rather than unblock the command stream. Only the final postprojection-Z residual completes the host call. Keep private Z live; normalized storage is reusable only after both up/gate consumers finish. Model borrowed buffers and all joins in the lifetime/resource IR, then check linked memory.

Do not assume that the computed attention tensor obeys a supplied-input bound. Derive its bound from probability mass/value bounds and validate original-input mathematics. Require **separate** score/softmax witnesses, attention-output accuracy (existing 2% L2/2.5% peak contract), output-projection accuracy, MLP-delta accuracy and final residual accuracy (existing tail 2%/3% contracts), plus immutable inputs, repeated warm calls, zero/nonuniform-gamma/cancellation cases, corruption rejection, matched source controls and counter-versus-sampled overhead. No silent weakening of a branch gate to make the composition pass.

Only after that boundary is qualified, connect preceding input RMS and Q/K/V projections and the two source RoPE operations with explicit layout/frequency conventions. Head replication, masks, cache ownership/update and complete Prefill/Decode follow as separate source-backed work; the 23-node boundary does not cover them.


### 31-node input-attention boundary qualified — 2026-09-07

The mixed 64/F256/8×8 counter profile is registered as configuration 134. Eight
SDK calls, 85 fault rejections, thirteen actual native branch observers on two
hosts and 23 source/HLS port groups across eight calls pass. The matched adapted
source max-PE cycle overhead is 1.14–1.22%; static memory is 35,952 bytes per PE.
The failed all-half cancellation configuration is retained and unadmitted.

Next remain linear algebra: source-backed Decode's batched vector/matrix and
axis-reconfigured collective chain, then model-level head/mask/cache semantics
where implementation evidence exists. The inspected Decode source reads externally
supplied XKCache/XVCache and does not append computed K/V into them. Its
pes_p_head/pes_p_kv_head parameters are declarations without uses in decode.csl.
Do not label a port as cache update, GQA or complete autoregressive Decode merely
because those names appear in configuration. Establish actual sharding, host input
packing, reduction groups and repeated-call state before selecting that boundary.
Larger mixed resident shapes must first pass typed memory/range checks; 64-only
admission does not imply general geometry support.

### Batched Decode normalized fanout — 2026-09-07

RMS B3/N512 and normalized three-branch QKV B3/N512/F512 now qualify as135/136.
QKV8standard and8source calls match all10 raw groups,48faults rejected, two-host
native CSL identical, maxPEcycle overhead0.5604–0.5606%, static37168B/PE.
UP/GATE B5/N256/F512 is the next running configuration; shared structural SSA
canonicalization, local DSR matmul, padded dynamic-length Y collective and
complete-call audit are reused. See `docs/BATCHED-PROJECTION-FANOUT.md`.
Before a composed down/cache path, settle axis ownership and SDK completion
contracts in `docs/DECODE-AXIS-HANDOVER-DESIGN.md`; no local-return barrier
assumption is admitted. Preserve explicit SDK math versus Decode fast-exp policy.

UP/GATE subsequently qualified as137:8standard/8source calls,9exact port groups,
44fault rejections, maxL2 .00629146, cycle overhead0.7736–0.7740%, static20672B/PE.
The next work is isolated skewed axis-handover execution before selecting a
composed cache/down path; the control-ring baseline is not a production choice.

Current item8 follow-through (2026-09-07): resident B5/N256/F512 full FFN is now
source-expressed and native-gated at every stage. Generated SDK CSL is executing
at `projects/waferllm/batched_feed_forward_5x256x512_8x8/run-20260907T200435840912Z`.
Prefer independent SDK X/Y f32 collectives with explicit half local policy over
the costly experimental ring handover. Full8 calls, original vecmat compute
control, frozen fault audit and admission remain required. The debugger/shared
runner integration has its own native-only202543 bundle with identical CSL;
qualification count remains137 until the device acceptance gates finish.

Current status superseding the development note above: full FFN is admitted as
bounded profile138, with eight matched SDK/control calls,52 mutation rejections,
full seven-stage original-math gates and static memory evidence. See
`docs/BATCHED-FEED-FORWARD.md`. Next, continue the missing Decode cache-attention
contractions/softmax/output tail using explicitly read-only supplied cache inputs;
do not add another FFN shape in place of those missing algorithm semantics.

Cache-attention boundary now admitted as139: B5/N256/S512, eight HLS/control calls,20 exact groups,67 rejected faults, independent five-stage gates, two native hosts and linked memory. See `docs/SUPPLIED-CACHE-ATTENTION.md`. Continue source-backed Decode batched pair rotation and its Q/K layout contracts before wider QKV/cache composition. Existing bounds and supplied read-only cache semantics must remain explicit; no full head/GQA/cache-update claim.

Decode-layout pair transform now admitted as140: B5/N1024, feature X partitions/Y replicas, explicit source odd_even order,6HLS/repaired-source calls,30fault rejections,2native hosts and linkedmemory. Actual source odd-offset reset defect and8callDSD/alias probes are preserved; see `docs/BATCHED-PAIR-ROTATION.md`. Next connect normalized QKV/pair/read-only-cache attention through one device graph, with numeric bounds and SDK→local DSR leases; do not claim full Decode from these separately qualified stages.

### Active item8 graph: normalized QKV → pairs → supplied cache (2026-09-08)

The25-node B3/N256/S512/8×8 graph is qualified after eight actual SDK calls
and an original Decode vecmat local-compute control (qualification010554710275). It uses Q4
and K/V32 half blocks after the unchanged native cancellation gate rejected Q32.
Three outputs preserve new rotated K/projected V without appending them. Eleven
mathematical stages, actual native observers on two hosts, per-PE raw witnesses,
finite range/lifetime plans and static ELF measurements are available. Full-run
source comparison,107corruption rejections and admission as bounded configuration141
are complete. HLS/source cycles65606/66854; static46592/47744B perPE.

After this graph, recover the next source-backed boundary explicitly: connect
the attention result to the already qualified normalized FFN while respecting
its original residual and weight/layout semantics. A single-PE-region SRAM
budget must be proven before claiming this composition; separate regions or
workspace reuse require explicit ownership and communication, not hidden host
intermediate computation. Full Decode heads, masks, position generation and
cache updates remain separately unimplemented semantics. Stay in the numerical
queue; do not advance stencil or physics on the strength of this partial chain.

The next FFN boundary now has an analysis-only counterexample in
`evidence/projected-cache-ffn-next-boundary.json`: legal coherent attention inputs
produce predictedZ=33, whose next half RMS sum overflows. Naive sampled allocation
plus FFN weights alone is59,104B/PE. Resolve statistic precision/scaling and
resource composition before implementing a complete chain. This witness is not
an extra SDK-qualified case and does not change the active eight-call fixtures.
