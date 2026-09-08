# Resident batch-major FFN over SDK collective planes

The B5/N256/F512,8×8 configuration is qualified in
`evidence/qualification-20260907T210527940077Z.json` (bounded profile138).
Its HLS source is `projects/waferllm/batched_feed_forward_5x256x512_8x8/hls.cpp`.
This is a full RMS→UP/GATE→SiLU→product→DOWN→residual numerical subgraph, not
full Decode, KV-cache management, an entire source project or hardware evidence.

## Source and precision contract

The algorithm and local projection recurrence follow WaferLLM Decode commit
fd1c2daae37cd68706c03fc8009887ecee9900f8, `Decode/src/decode.csl`. Existing source
RMS recurrence/input defects are repaired. Stable-sign SDK half SiLU replaces
its fast_exp approximation. These differences are explicit, not hidden under an
unmodified-source claim.

Local arithmetic is binary16, including stationary-memory DSR RMS accumulation
and row-major matrix FMAs. Communication explicitly widens to f32, invokes SDK
reduce_fadds+broadcast, then narrows once. Independent X/Y planes avoid changing
the axis of live data routes. Native C++ uses standard RMS/SiLU mathematics and
its declared half intermediates. Native and device each satisfy fixed2%relative
L2/3%peak-scaled stage checks against original inputs; they need not be bitwise
identical to each other. The matched original vecmat compute control is bitwise
identical to device observations.

## Physical program

| Stage | Storage/distribution | Completion |
| --- | --- | --- |
| Input and gamma | Model features sharded Y, replicated X | Host inputs before launch |
| RMS | Local half squares/sum; SDK Y f32 reduce+broadcast | SDK broadcast callback, then normalization |
| UP/GATE | Resident weights[Y-input,X-hidden]; packed branch/batch/output | One fused Y collective |
| Activation/product | Hidden features X, replicated Y; raw gate preserved separately | Synchronous SDK math/map and half multiply |
| DOWN | Resident weights[X-hidden,Y-output] | Local DSR FMA, then SDK X collective |
| Residual | Original input retained; output features Y, replicated X | Final local add, then command-stream completion |

SDK X uses colors0/1, queues2/4 and local tasks14/15; Y uses colors4/5, queues3/5
and tasks16/17. Caller continuation uses local task10. Explicit DSR set1/2 leases
are released only after the relevant synchronous operation or SDK callback.
Callbacks are local operation completion, not a claim of mesh-wide quiescence.
The default SDK memcpy reservations are checked separately.

The numeric/observation ledger plus protocol/code/stack reserves totals40676B/PE.
Linked HLS static footprint is38400B/PE across9ELF classes; the control is38768B.
The10752B static remainder is not a measured dynamic stack high-water mark.

## Executed evidence

- SDK bundle: `run-20260907T200435840912Z` under the project directory, eight calls.
- Original vecmat control: `evidence/batched-ffn-source-compute-20260907T201155253854Z`, eight calls.
- Comparison: `evidence/batched-ffn-full8-source-compute-comparison.json`.
- Fault rejection: `evidence/batched-ffn-full8-mutations.json`,52 rejected corruptions, including late/stale calls.
- SDK-host C++: `evidence/batched-ffn-native-host-2013`,365 provenance files verified, six actual intermediate observers and identical CSL.
- Static memory: `evidence/batched-ffn-static-memory-2004.json` and the source-compute counterpart.

All15 compared raw groups, including progress, match across all PEs and calls.
Queue masks and timestamps are checked separately. Seven original-input stage
gates pass per call. Worst device relative L2 is0.0033915367014 at DOWN; final
output maximum is0.0008858708350. Six actual native observers are checked on both
hosts so a large residual cannot mask an incorrect DOWN increment.

Max-PE device intervals are48491–59696 cycles. HLS uses250 fewer cycles per case
than the original vecmat control: ratio0.994870848–0.995829580. The control retains
the same SDK collective, repaired RMS, stable SiLU and observation schedule;
this measures local projection lowering overhead, not unmodified Decode speedup
or hardware throughput. Full diagnostic host calls take258.6–347.0seconds in this
simulator run, including every weight/input/output readback. These times are
separate from the PE cycle intervals. Derived f64 norm summaries vary in their
last bits across host NumPy/BLAS implementations; raw half words and integer
cycle checks remain exact and both hosts independently meet the fixed gates.

## Shared middleware improvements

`feed_forward_ir.py` canonicalizes the13-node graph independently of names or
layout. The older tiled FFN uses the same structural verifier. The new physical
profile adds explicit precision, ownership, SRAM and callback-bound lifetimes.
The reusable CSL modules are `sdk_axis_reduce.csl`, `sdk_stable_silu.csl`,
`batched_rms_local.csl` and `batched_matmul_local.csl`.

The stable SiLU primitive was checked over all63488 finite half encodings. Its
maximum absolute standard-math error is0.00417110662046 and maximum rounded-half
ULP difference is9. There are193 nonzero rounded standard results that become
zero, including negative tails and near-zero rounding. Application stage gates
remain necessary. Directed SDK association tests also distinguish reverse-linear
f32 reduction from forward order on8512 observed words; earlier exact-dyadic
probes alone did not establish that order.

`rms_l1_bounds.py` retains normalization correlation using Cauchy-Schwarz and
explicit rounding/underflow envelopes. It preserves the independent elementwise
finite-value proof. For the current shape, normalized L1≤258.629 tightens the
projection bound from128 to8.21875. Fractions and an outward square-root enclosure
avoid understating the bound. Fresh native205552 passes all sealed gates and
emits identical CSL to the qualified SDK bundle. All40 actual SDK rows satisfy
the exact-rational inequalities in `evidence/batched-ffn-l1-full8-review.json`.
This tightens analysis; it does not claim SDK qualification for new input bounds.

The shared runner supports `batched_ffn:5:256:512`, six native observations and a
separately sealed target preflight. Debugger steps0..9 inspect actual local/reduced
RMS, normalization, local/reduced UP/GATE, activation, product, local/reduced DOWN
and residual. Partial-call checks remain distinct from full process qualification.
The latest complete regression suite passes289 tests.
