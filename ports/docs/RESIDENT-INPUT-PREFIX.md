# Resident input RMS, Q/K/V projections and attention chain

The explicit mixed-precision 31-node 64×64/F256/8×8 profile is now qualified through the regular compiler and SDK 2.10.1 driver. Its public HLS source, thirteen actual native observers, eight SDK calls, original-eleven-input branch checks, 85 corruption rejections, 23 exact matched-source port groups and linked ELF memory are recorded in `evidence/qualification-20260907T152814610270Z.json`. The source/HLS simulator max-PE cycle ratio is 1.011368–1.012161 (about 1.14–1.22% composition/observer overhead, excluding host I/O). Static memory is 35,952 bytes/PE. This is not full Prefill/Decode or hardware throughput.

The plain-half 31-node variant remains rejected after the cancellation failure. Sections below retain the development chronology and failed experiments; earlier descriptions of pending admission refer to those checkpoints. Current mixed source: `projects/waferllm/input_attention_mixed_64x64x256_8x8_counters/hls.cpp`.

## Source semantics to preserve

Pinned local reference `projects/waferllm/upstream/Prefill/src/prefill.csl` schedules `rmsnorm_x`, `xq_matmul`, `xk_matmul`, `xv_matmul`, `xq_rope`, `xk_rope`, then the score/softmax/value/output/FFN tail. Its `z_add` adds the original X input to the output projection. Its two RMS stages use the same `W_tile` feature vector. The initial boundary should represent these shared SSA inputs directly rather than introduce unrelated residual or normalization operands. A separate-gamma extension must be labeled as such.

For that source-shaped boundary there are11 public inputs: X, gamma, Q/K/V weights, supplied cosine/sine tables, output weight and up/gate/down weights. Six preceding operations (input RMS, three matmuls, two pair transforms) plus the23-node chain's replacement input set give31 total typed nodes, with one final output. This count is descriptive, not a dispatch based on application names.

Q/K/V projections share the same normalized X and its initial left alignment. The qualified normalized-fanout experiments already establish that a subsequent branch must consume the live left buffer with the correct pointer parity. Do not restore pointers to originally named buffers after a completed shift. Source Q/K/V weights are destructive; the HLS contract should keep public weights immutable and copy to shared working storage at phase entry.

Source `xq_rope` and `xk_rope` compute `(odd*cos-even*sin, even*cos+odd*sin)` with coefficients broadcast over token rows. The explicit frontend operation is `rotate_pairs<pair_order::odd_even>` with `coefficients=feature_pairs`, not silently the standard even_odd transform. Existing SDK source probes show that zero angle swaps adjacent supplied columns. They also show the source temporary-vector length bug when local token rows differ from local_features/2; the shared local implementation must size all four temporary vectors by token rows. See [PAIR-ROTATION.md](PAIR-ROTATION.md). Coefficient generation, token positions and model head packing remain separate unsupported model concerns.

## Resident handoffs

The original X must stay live for the first residual add. Its physical input can remain the existing residual allocation; the current Q/K/V public allocations become private projected/rotated values. Preserve raw Q/K projection snapshots for independent pair-transform auditing before in-place rotation. Normalized X is shared by all three projections; release it only after V completes. Q and K pair transforms use synchronous row-length memory DSDs and no new application routes/tasks. Join the V projection and both transforms before the score phase consumes Q/K/V.

Use one shared task/phase dispatcher. Do not launch a second host region or regenerate computed Q/K/V from Python. Reuse existing projection weight/receive storage where extents permit. Keep the current23-node entry descriptor reset for contiguous QK contraction and the existing PV-to-output-projection reset. Extend lifetimes and actual ELF checks before admitting a profile.

## Range and numerical obligations

Computed Q/K/V are not arbitrary supplied tensors with a guessed bound. Derive normalized feature bounds and half projection bounds, then account for the two half products/addition of each pair transform. A useful stronger bound may use the correlated row norm rather than multiplying the maximum feature bound by N; it needs an explicit rounding argument. Reject unsupported ranges before generating CSL. Do not shrink a difficult fixture or weaken a branch threshold to make the connection pass.

Independent mathematics must start from original X, weights, gamma and supplied tables, and separately check normalized X, raw Q/K/V projections, both pair transforms, score/probability/mass, attention, output projection, delta and final output. Pair transforms retain a cancellation-aware product-magnitude error contract rather than an unreliable output-relative tolerance. Repeated SDK calls must change inputs/tables, exercise zero and nonuniform-gamma cases, and prove that the source's branch ownership and descriptor resets survive the full connection.

The present23-node evidence proves none of these missing preceding computations. Preserve the31-node distinction from masks, replicated heads, KV-cache ownership and full Prefill/Decode, even once this numerical path executes.


## Preparation checkpoint

`experiments/input_attention_source.py` now produces a31-node C++ proposal using the existing typed operations and pragmas. A Clang frontend test checks eleven public inputs, shared gamma, original X residual, shared coefficient inputs and explicit odd_even broadcasts. At that preparation checkpoint, semantic dispatch rejected the proposal. The later diagnostic generation/execution is recorded below; regular dispatch still rejects it and no31-node qualification is claimed.

`toolchain/rms_projection_bounds.py` supplies an unselected range certificate for the next lowering. It bounds a normalized row's L1 norm using its rounded-square sum, a conservative positive-reduction depth, Cauchy–Schwarz, gamma-first rounding and exhaustive SDK inverse evaluation. Underflow terms and the high-precision native RMS path are separate. Ordinary half projection bounds then cover any K permutation. For N64, input magnitude.125, gamma1.5 and weight.00390625, the outward row-L1 bound is98 and projection bound.395263671875. This avoids the much looser featurewise product bound (~2.996), without changing a fixture or numerical tolerance. Four tests cover zeros/rejection, sparse and uniform underflow rows, target/native arithmetic, and all eight normalized outputs of a previously qualified real SDK FFN bundle. This is a model-conditioned range certificate, not an executed new prefix.


A separate adapted-source control `evidence/input-attention-source-20260907T122338297913Z` now connects the actual preceding source stages to the whole existing numerical tail. Its three-case predicted-target/original-eleven-input preflight passes (`input-attention64-source-preflight.json`); All three SDK calls and full source arithmetic/original-input reviews have now passed (`input-attention64-source-executed-review.json`). This is an executed source control, not HLS31 qualification. It retains source-stage snapshots of input normalization, raw Q/K/V and rotated Q/K before destructive downstream use. The adapter repairs input RMS indexing, preserves live normalized-X pointer roles for K/V, and sizes all pair temporaries by local token rows. These edits apply only to the fresh source control. The later diagnostic lowering and its precision boundary are described below.


The unselected `mesh_input_attention.py` plan now verifies the31-node structure, derived Q/K/V ranges, shared gamma/X residual and source pair conventions. It composes the23-node child with25 declared lifetime phases (`input_attention_lifetimes.py`). Dead normalized storage holds the four row scratch vectors only after V joins and before score setup. Planning admits64/F256 and128/F256 counters and64/F128 sampled under the existing reserve, rejecting oversized full sampling and unsupported derived Q/K ranges. Four structural/range/alias tests pass. Regular compiler dispatch and final SDK qualification remain pending. The shared generated diagnostic path is described below.


## Executed shared generation and precision boundary (13:18 UTC)

The development profile lives in `projects/waferllm/input_attention_64x64x256_8x8_counters`. `mesh_input_attention.generate` composes the existing engine via hooks; `pair_rotation_local.csl` implements synchronous in-place row-vector rotation with four borrowed row scratch vectors. No extra fabric routes or local tasks are allocated. `mesh_input_attention_sdk.py` declares eleven-input transport and all diagnostic extents. Its packed first three cases exactly match the executed source control; original inputs stay immutable.

Frozen generated bundle `evidence/input-attention-codegen-20260907T130338205990Z` compiled in SDK2.10.1 and completed three calls. `input-attention-codegen-source-review-1312.json` checks 17 half/f32 port groups bit-for-bit against the adapted source, original-eleven-input mathematics and prefix progress. Local HLS/source maximum-PE intervals are141604/141396,141607/141397,132639/132435 (about0.15% overhead). ELF high water29840B/PE, remaining19312B. This is diagnostic schedule/resource evidence, not numerical qualification of all inputs or hardware performance.

The actual native eight-case gate fails case3. The same unchanged case executes in SDK bundle `input-attention-codegen-20260907T130756542610Z`; `input-attention-sdk-cancellation-1314.json` records the failure. Original mathematical projection is-.1256540144; actual native and SDK projection is-.1251220703. With X=.125, mathematical Z=-.0006540144 but actual Z=-.0001220703. SDK delta relativeL2=.95578093 and final=.83062258. Early-stage small relative errors become large after residual cancellation and RMS normalization. No fixture, epsilon or tolerance was loosened.

Actual native precision study `input-attention-precision-20260907T130605526102Z` replaces five ordinary contractions by explicit block-f32 merges; block8 and block1 both still fail final relativeL2=.07114369. The later labeled study `input-attention-mixed-precision-20260907T131046512625Z` preserves actual Z and normalized-Z outputs. Its helper functions are experimental C++ arithmetic, **not registered frontend/backend features**:

| Retained f32 region | Case3 final relativeL2 | Eight native cases |
|---|---:|---|
| O projection/Z | .65209060 | fail case3 |
| PV/O/Z | .07114369 | fail case3 |
| V/PV/O/Z | .00640749 | all pass |

This does not establish the corresponding CSL policy. `input-attention-probability-sensitivity-1317.json` starts from actual SDK input normalization/logits/probabilities, then performs **f64 mathematical sensitivity calculations**, not predicted CSL execution. Observed half probability mass error.0004272461 still leads to Z relativeL2=.05475395 even when V/PV/O/Z arithmetic is widened in that study. Mathematically normalizing observed logits reduces Z relativeL2 to.00614797. Therefore the target precision design must address probability normalization/representation as well as V/PV/O/Z.

Next work: explicit storage, accumulator and fabric precision contracts; f32 probability normalization and connected V/PV/O/Z representation; mixed-width conversion/DSD/communication handling with joined ownership; updated derived bounds, lifetimes and actual ELF. Preserve half Q/K and other stages where validated. Run all eight actual native and actual SDK branch/final gates before registration; performance comparisons must identify the precision difference. The catalog remains133 qualified profiles.

Preserved development failures:130055476644 passed native execution but its new gate adapter passed a list instead of the required output dictionary;130117743024 exposed the real cancellation gate failure. Neither executed SDK. The first source-comparison script reversed source `result`/`output` semantics for the delta mapping; its failed read-only review log is retained. Corrected names match the established17/23-node comparison; no computed values or tolerance changed.


## Explicit mixed frontend and standard pipeline (continued)

The full half failure remains unchanged. The actual public precision overloads
now express f32 V, probabilities, PV, output projection, Z and post-Z RMS,
with explicit half storage before MLP and at final output. Clang retains all
seven boundaries. Native interpretation agrees exactly with all eight actual
C++ results and all 13 observed branches.

`mesh_input_attention_mixed.py` is selected structurally by the regular compiler.
It checks the 31-node chain, shared original X/gamma, explicit boundaries,
64x64/F256/P8 counter geometry, mixed transport and phase leases. Its half
shadow only reuses topology/layout restrictions. Mixed ranges, physical byte
lengths, storage dtypes and ownership are separately described. The estimate is
42,258 bytes/PE including conservative reserves; executed mixed ELF high water
is 35,952 bytes/PE, leaving 13,200 static bytes. This is not a dynamic stack bound.

The pure shared continuation emits twelve CSL files identical to the executed
mixed prototype. The first direct typed-driver bundle is
`evidence/input-attention-codegen-20260907T142401616614Z`: actual C++ → Clang →
checked typed IR/plan → shared CSL, with no CSL adapter. All eight SDK calls and
all original-eleven-input mathematical gates pass. The current-plan review is
`evidence/input-attention-typed-review-20260907T144222608116Z`; it explicitly
separates updated plan/auditor metadata from the earlier frozen SDK bundle.

The shared target auditor checks the half prefix RMS/QK/pairs/score exactly,
98,304 V/PV/O binary32 output words in distributed FMA order, 294,912 MLP f32
accumulator values, residual additions/narrowing, immutable inputs, roots,
completion counters and owned queue drain. Softmax and RMS accuracy at observed
producer joins are separate from original-input application mathematics.
Mutation tests cover 30 physical ports plus lifecycle/output and word-range
faults. An initial queue test flipped an unowned status bit and correctly was
not rejected; the corrected test flips an owned queue bit. Logs are preserved.

The regular compiler fresh build is
`projects/waferllm/input_attention_mixed_64x64x256_8x8_counters/run-20260907T144456167691Z`.
Actual native outputs and 13 separately compiled observers pass. Its preflight
uses independently rechecked **prior actual SDK** data with identical HLS,
CSL and fixtures; this is explicitly a replay gate, not a predicted arithmetic
model. Its standard-driver SDK execution and catalog/performance qualification
must be reported independently; do not infer completion from the prior run.

For the math range and preserved wider exp-domain failure, see
`F32-PRECISION-BOUNDS.md`. For all these kernels, performance is currently WSE3
simulator evidence, not CS-3 hardware or full-model throughput. The mixed policy
adds communication/computation versus the failing half policy; it is not a
speedup claim and still needs a precision-matched source performance review.
