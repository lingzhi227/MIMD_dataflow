# Inference stages: source reconnaissance after bounded contractions

This is preparation for queue item8 while FFT validation runs; no inference implementation or qualification is claimed. Pinned WaferLLM commitfd1c2daae37cd68706c03fc8009887ecee9900f8. Existing matmul/GEMV profiles are dependencies, not proof of full inference.

## Numerical semantics must come from code and execution

Prefill `rmsnorm_x` and `rmsnorm_z` square in half, sum locally in half, invoke row communication, then divide the result by `head_dim`. Do not silently rename this as the usual hidden-dimension RMSNorm until host dimensions and reduction ownership establish the denominator. Weight multiplication and inverse-norm multiplication are distinct half rounding points.

Prefill softmax scales scores with alpha, initializes max to zero, allreduces max, subtracts it, calls SDK `math.exp_f16`, allreduces half sums and multiplies by a reciprocal. Initializing max to zero may be numerically relevant for all-negative scores and underflow. It is not a reason to change the source without a separate math/reference contract.

Decode `fast_exp` atlines144–151 computes `1+x/256`, then squares twice: mathematically `(1+x/256)^4`, with half rounding. This is not the usual exponential or the common eight-squaring approximation `(1+x/256)^256`. Both Decode softmax and SiLU call it. A faithful source port must expose this explicit numerical policy and measure its discrepancy from standard functions; a standard softmax/SiLU must not inherit it invisibly. No source bug conclusion or upstream message has been made; execution and host expectations must be examined next.

Prefill preserves shifted X_norm/Z_norm across Q/K/V or FFN contractions. Its controller advances phase flags through communication completion callbacks; restarting each contraction independently would discard intended reuse. Decode includes K/V cache input ownership, batch axes and grouped reductions. The existing fixed-host-pointer and per-phase buffer ownership rules must extend to these stages.

## Required next evidence

1. Preserve pinned host packing, config and native reference scripts; inspect what is actually checked, including repeat behavior.
2. Probe SDK half sqrt/exp and source polynomial behavior without replacing them by Python values. Separate target arithmetic, source-equivalence and standard-function accuracy.
3. Express distributed norm/max/sum, layout transitions, cache state and contraction reuse in typed IR with explicit precision and callback/buffer lifetime.
4. Start with individually qualified distributed numerical stages, then a bounded complete prefill/decode graph. Preserve original and altered policies as distinct contracts; do not claim model quality from numerical kernel completion.

## Pinned host inspection

Host/config files are now preserved under `references/waferllm-inference-host`, extracted with git show at the pinned commit; none was executed. Both supplied simulation configs use P8,dim64,head_dim64,n_heads1 and sequence64. The provided larger Prefill config also sets dim=head_dim4096,n_heads1. Thus the denominator concern above does not create a mismatch for these supplied configurations; general multi-head semantics remain unestablished.

Both launch_sim scripts copy back timing buffers and report cycles; they do not gather and compare final inference tensors. A successful run of those launchers therefore is not numerical qualification.

Decode launch_sim passes `(repeat_steps,warmup_steps)=(1,0)` to `decode_host`, whereas the pinned CSL signature is `(total_warmup_times_,total_repeat_times_)`. Source reading indicates one untimed warmup with zero timed repeats; the start timestamp condition is skipped before exit. This requires a source-preserving argument-order adapter and real execution before a meaningful baseline. No benchmark from that unadapted launcher is accepted as performance evidence.

## Additional source-level discrepancies requiring execution probes

These are observations of the pinned code, not yet device-confirmed numerical conclusions. They must not be silently inherited by a standard RMSNorm operator.

- Decode `rmsnorm_x` first stores X² in X_tmp, then sets the weight-multiplication input descriptor to ptr_X_tmp (around lines369–377). The source therefore appears to weight the squared input rather than X. Signed inputs will distinguish the behaviors.
- Prefill `rmsnorm_x` uses a length-seq_len_p_pe descriptor for each feature column. Its final loop over dim_p_pe multiplies each whole column by scalar local_sum[i], although local_sum has seq_len_p_pe entries. The intended per-row normalization would instead use the vector of row factors. Distinct row norms detect the difference even when local sequence and feature dimensions happen to match; unequal dimensions also require a bounds review. Do not use square/uniform fixtures as sole validation.
- The pinned Prefill head_dim constant is explicitly dim_p_pe*P (line13), whereas Decode derives it from head_dim_p_pe*P. Keep these source definitions separate from host configuration names.

The next stage should execute isolated, source-preserving adapters with sign-varying inputs and distinct row norms, alongside separately specified standard RMSNorm. A source-equivalence result is not automatically a correct standard-function result. The untouched upstream tree remains the reference.

## Primitive execution checkpoint

`evidence/inference-math-20260906T205105818691Z` executes three changed/reversed/zero calls in SDK2.10.1. All54source polynomial values agree bit-for-bit with sequential half `(1+x/256)^4`. SDKexp_f16 andsqrt_f16 have zero nearest-half ULP distance on these selected inputs, including signed zero/subnormal transport; this is not a universal rounding guarantee. Atx=10 the source polynomial differs from mathematicalexp by about22025.3, so it cannot be used invisibly for standardexp. Raw bits and quantitative checks are in `inference-math-analysis-20260906T2059Z.json`.

A further host-layout concern: Prefill launch_sim constructs W using `np.tile(W.reshape(P,dim_p_pe), reps=(1,P))`. Under its ROW_MAJOR PE transfer this repeats the weight segment selected by PErow, whereas X feature ownership is by PEcolumn. An all-one W isolates the earlier normalization-factor issue; a later nonuniform-weight test must distinguish original host packing from the declared logical weight layout. This remains source-level reasoning until tested.

Pinned git show hashes for Prefill comm_pe.csl and Decode decode.csl exactly match the read-only local copies; the pinned repository is clean. The observations above refer to that recorded commit, not a changed local source.

## Distributed original RMS execution checkpoint

The source-preserving64×64/P8 probe `prefill-rms-20260906T210620408155Z` passes three changed-input SDK calls. Only its continuation becomes host completion; a reduced-sum observation and result/inverse exports are added. Arithmetic and communication source remain unchanged, with all-one weights isolating scaling from weight-host-layout concerns.

Source comments about root receive order were insufficient: mv_left_recv is physically routed from EAST. A left-first reference mismatched5logical row sums by1ULP (replicated40half words); its analyzer and failure are preserved. The route-derived right-first reference matches every global reduced half value exactly.

Final outputs exactly match multiplication by the *observed feature-indexed* inverse factor. Against mathematical row-wise RMSNorm, relativeL2 errors are0.05917for random input and1.98505for proportional rows with distinct norms; zero remains exact. This is successful source execution, not standard RMSNorm qualification. The nominal sequential-nearest-half inverse model also differs at selected values, by up to2half ULP in the random case; the earlier18-point primitive probe was not a universal rounding guarantee. Further sqrt/divide observation is required for a precise target numerical policy. See `evidence/prefill-rms-source-analysis.json`.

## Exhaustive half math qualification (2026-09-06 21:51 UTC)

`evidence/rms-math-20260906T212318272243Z` completed all62warm calls covering31,744nonnegative finite binary16 encodings. `evidence/rms-math-exhaustive-analysis-20260906T2151Z.json` verifies exact source-derived sqrt bits, nearest-half reciprocal of the observed roots, staged and whole-expression RMS inverse, ordinary split multiply/add and explicit DSD FMA. SDK sqrt is non-nearest for6,375inputs, always within1ULP of nearest; split/fused differ for20,832probe inputs. SDK `inv_f16` agrees with language division over these observed roots, not a universal reciprocal-domain proof. Negative zero and special values are outside this exhaustive set.

The initial scalar-pointer `@fmach` compile failure remains in211916134945. The successful probe uses three one-element DSD operands, satisfying the compiler's intrinsic signature. This is target-math qualification, separate from the distributed HLS stage's application-level numerical and protocol audit.

## Original Prefill softmax execution

Source-only `evidence/prefill-softmax-20260906T215552252478Z` completed three warm calls with the original arithmetic and communication. Analysis `evidence/prefill-softmax-source-analysis.json` finds:

- Random score inputs: standard stable-softmax relative L2 error 0.000541947.
- Every input score −1024, with source alpha 1/sqrt(64): maximum stays zero because of its initializer; scaled inputs remain −128, all SDK exponentials and reduced sums become zero, reciprocals become +Inf, and all 4096 output entries are NaN.
- The following all-zero call recovers to exact uniform 1/64. This separates the numerical issue from stale protocol state.

The source was not repaired for this experiment. A standard HLS softmax must initialize its maximum from valid data or a true lower bound, then preserve the max-shift, exponent, sum, reciprocal and vector-scale stages. Decode's fourth-power `fast_exp` is a different approximation policy; it must not silently replace Prefill's SDK exp or the standard mathematical contract.
