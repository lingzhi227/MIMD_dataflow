# Supplied-cache attention and output residual

This source-backed port extends the linear-algebra queue with the supplied-cache section of WaferLLM Decode (`fd1c2daae37cd68706c03fc8009887ecee9900f8`). The current application is `projects/waferllm/cache_attention_5x256x512_8x8/hls.cpp`: five queries, 256 features, 512 cache positions on an 8×8 PE region.

This boundary consumes queries and keys after any desired rotation, one shared read-only K/V cache, an output weight and a residual. It computes QKᵀ, stable unmasked softmax, PV, output projection and residual. It does not append to the cache, select heads/GQA groups, apply a causal mask, generate queries or execute a full model. These are distinct semantics, not inferred from a common matrix equation.

## Frontend and physical dataflow

The HLS uses the existing `matmul_blocked<f16,scalar>` operation: half FMA partials with a float merge. Its block sizes are 32, 32 and 32 for score, PV and output. On the current fabric, score/output have 32 resident inner features. PV has 64 resident sequence positions and performs two local blocks before the SDK reduction. The dataflow declarations select resident operands, the reduction axis, output ownership and precision explicitly.

| Object | Logical shape | PE(y,x) ownership / local order |
| --- | --- | --- |
| query | B×N | feature X, replicated Y; batch-major |
| K | S×N | sequence Y × feature X; local feature-major `K[y*St+j,x*Nt+i]` at `i*St+j` |
| V | S×N | same logical block; local sequence-major at `j*Nt+i` |
| score/probability | B×S | sequence Y, replicated X |
| context | B×N | feature X, replicated Y |
| output weight | N×N | input feature X × output feature Y |
| residual/output | B×N | feature Y, replicated X |

The key transpose is an input view implemented by serialization, with no host contraction or device transpose. The recovered upstream Python key packing disagrees with the device's feature-X score contraction. The port uses logical indexing and device source semantics, including non-square dimensions and coordinate-directed fixtures; recovered host packing is not used as its oracle.

`attention_ir.py` canonicalizes the existing supplied-Q/K/V attention graph. `cache_attention_ir.py` composes that structural graph with the output/residual edges. Verification is independent of variable names and declaration order. Placement policies, finite arithmetic bounds, physical buffers and callback-scoped DSR leases remain separate from numerical fixtures and validation.

## CSL facilities and resource ownership

The region uses the SDK's two independent `<collectives_2d/pe>` instances:

- X: colors 0/1, SDK tasks 14/15, queues 2/4, DSR bank set 1.
- Y: colors 4/5, SDK tasks 16/17, queues 3/5, DSR bank set 2.
- SUM completion task 10; MAX helper completion task 11. Default memcpy resources are checked separately.

`sdk_axis_reduce.csl` widens half I/O to float, calls SDK reduce_fadds and broadcast, then narrows. `sdk_axis_max.csl` uses gather and broadcast through caller-supplied functions accessing the **same** SDK instances; it does not import a second conflicting copy. MAX has `4*(P+1)*capacity` bytes of private storage. The public helper checks its index/extent capacity at compile time.

The controller serializes score X SUM → Y MAX → exponent Y SUM → context Y SUM → output X SUM. Provider callbacks release local workspace and DSR ownership before local computation or the next operation. They are not advertised as a global barrier. Inputs remain immutable until host readback. Odd batch transport padding uses -65504 for finite MAX and zero for SUM; only the five logical rows normalize.

Local softmax uses a true first-element maximum, stationary accumulator DSRs, SDK exp_f16 on nonpositive differences, and half reciprocal/normalization. Local projection uses the shared packed half DSR matmul. PV merges its two half partials in float. SDK 2.10.1 rejects the attempted mapped merge/narrow loop with an internal LLVM PHI validation error; an explicitly indexed merge/conversion compiles while preserving the packed half contraction. Its performance is measured, not assumed.

## Numerical gates and preserved failures

Every input case keeps the same fixed limits: 2% relative L2 and 3% peak-scaled error, independently for score, probability, context, delta and final result. Current authoring also requires nonnegative probabilities and row mass within 1%. Reference mathematics uses original inputs and stdlib fsum/exp; native intermediate values come from four actual C++ observers. A separate target model checks every exported PE word and the precise float collective association.

The fixtures include zero queries, signed data, coherent accumulation, all-negative logits, remote/different query maxima, sparse cache values, cancellation and coordinate-directed layouts. K/V and all other public inputs change across calls.

Two preflight failures are retained rather than weakened:

1. `run-20260907T214858391453Z`: ordinary consecutive half accumulation in native PV produced 4.872% context L2 error on the coherent case.
2. `run-20260907T215130589417Z`: shard-sized blocked native C++ passed, but the predicted half PV trajectory produced 2.174% delta L2 error on the cancellation case.

The explicit 32-element PV blocks pass both preflights with the same inputs and thresholds. `evidence/cache-attention-numerical-boundaries.json` binds the preserved failures. Finite bounds now follow actual half blocks → float merge → half narrow; a long half recurrence is not reused as an upper bound for blocked arithmetic.

## Current evidence and performance scope

Primitive evidence is complete:

- `evidence/sdk-axis-max-20260907T213834656626Z`: eight actual SDK calls, both MAX axes sharing SUM instances, all-negative maxima, skew, odd extents, immutable/out-of-place input and canaries.
- `evidence/sdk-softmax-math-20260907T213936783965Z`: all 31,745 finite nonpositive half encodings (including both zeros) for exp; all 9,217 half denominators in [1,512] for reciprocal. Exact model conformance and exp in [0,1], exp(0)=1.
- Compiler isolation evidence: `cache-compiler-unroll-20260907T215744841553Z`, `cache-compiler-isolate-20260907T215915966084Z`, `cache-block-compiler-20260907T220038587682Z`, `cache-block-loop-20260907T220201334092Z` under `evidence/`. These are compiler experiments, not application runs. The compact one-PE 2×4×2 reproducer `evidence/blocked-map-compiler-minimal-20260907T222242577114Z` also reproduces mapped failure/indexed success under the pinned SDK, with no simulator launch.

The full application is **qualified as bounded configuration 139** (`evidence/qualification-20260907T225536946434Z.json`). All eight warm calls passed on both SDK variants; all 20 observed groups match word-for-word. Five independent numerical stages pass on native C++ (two hosts) and actual device outputs. Worst device context L2 error is 0.008266679203, delta 0.005215773058, final 0.000206573928; maximum row-mass error is 0.004951477051. All 67 raw numeric/protocol corruptions were rejected. This is one admitted shape, not a full Decode implementation.

Every call measured 52,157 maximum PE cycles versus 52,975 for the original local-compute control (ratio 0.984558754129). These simulator results have the limited comparison scope described below. Full reports are `evidence/cache-attention-full8-source-compute-comparison.json` and `evidence/cache-attention-full8-mutations.json`.

Execution artifacts:

- HLS: `run-20260907T220401939966Z` under the application folder.
- Original local-compute control: `evidence/cache-attention-source-compute-20260907T220538494452Z`.
- SDK-host native rebuild: `evidence/cache-attention-native-host-2206`, eight cases and four actual intermediate observers; seven generated CSL files identical.
- Linked static high-water: HLS 36,640 bytes/PE, source control 37,168; nine ELF classes each. This is not dynamic stack measurement.
- Native authoring update `run-20260907T221440282260Z` raises planning reserve from 36,116 to 40,212 bytes/PE after linked-memory feedback, seals the shared-runner gates and generates the same seven CSL files. `evidence/cache-attention-planner-reserve-review.json` records this metadata-only update; it is not a second SDK qualification.

The compute control retains the original `gemv_static_step` and `vecmat_computation` bodies, with the same HLS local block merge, repaired softmax, SDK planes, inputs and observations. It measures local lowering overhead. It is not an unmodified Decode comparison, full-model benchmark or hardware throughput measurement. The three contractions total 3,276,800 FLOPs; the timing interval includes the entire resident pipeline, and diagnostic host I/O is reported separately.

## Debugging and reproduction

`experiments/build_cache_attention.py` uses the shared compiler and seals actual native and predicted-target gates. `experiments/execute_frozen_bundle.py` executes its immutable snapshot and audits each completed call; incomplete snapshots cannot qualify an application. The shared `run_ports.py` fixture is `cache_attention:5:256:512`.

Use `toolchain/debug.py <bundle> --node p7_7 --epoch 3 --step 4 --check-completed` to inspect an actual saved global maximum and audit the saved calls through the frozen implementation. Steps 0–13 cover local score, reduced score, scaled score, local/global max, exp, local/global sum, probability, local/reduced context, local/reduced output projection and residual. Unavailable calls and omitted counter-mode witnesses are identified explicitly, never filled from expected values.

All stage artifacts, native observers, compiler failures, per-PE raw words, resource plans and commands are retained in fresh directories. Qualification additionally requires full eight-call HLS/source comparison, linked ELF checks, two native hosts and mutation rejections through the frozen auditor.

### Half signed-zero fidelity

`evidence/blocked-matmul-signed-zero-20260907T225439920308Z` executes eight direct/blocked contraction probes using the qualified runtime. A final negative half underflow retains −0 on the direct DSR path, while the explicit float merge starts at +0 and yields +0. The target reference now selects the actual direct path; native blocked arithmetic is unchanged. `evidence/cache-attention-signed-zero-reference-review.json` reaudits all eight original device calls under the corrected model. No generated CSL changes or extra application qualification are implied. The initial syntax failure remains in `blocked-matmul-signed-zero-20260907T224857138657Z`.
