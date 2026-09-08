# CSL pragma syntax and extension boundary

A supported pragma annotates the next numerical statement. Attribute order is independent of meaning, and whitespace around `=` is allowed. The parser rejects duplicate, missing, malformed or unknown attributes; IR verification then checks types, shapes and policy compatibility, and the spatial planner checks resources.

```cpp
#pragma csl dataflow rows=4 cols=4 exchange=two_hop initial_align=bidirectional reduce=local overlap=double_buffer fp=relaxed compute=dsr
```

This is equivalent to:

```cpp
#pragma csl dataflow compute = dsr fp = relaxed overlap = double_buffer reduce = local initial_align = bidirectional exchange = two_hop cols = 4 rows = 4
```

`toolchain/pragma_contracts.py` is the declarative syntax registry. It canonicalizes recognized attributes before constructing IR. Existing 75 catalog sources produce exactly the same frontend IR as before this parser change; compatibility evidence is `evidence/pragma-compatibility-20260906T182003306427Z.json`.

## Supported directive families

- `place x=<unsigned> y=<unsigned>`: explicit placement for compatible actor operations.
- `resident`: compatible resident-state operation.
- `vectorize`: the qualified grid update vectorization transformation.
- `dataflow`: a supported numerical strategy with required layout, communication and precision attributes.

Dataflow strategies currently cover SDK collective GEMV/SUMMA, no-pivot blocked LU, Givens QR, lower right-looking Cholesky, CSC train SpMV and resident solvers, contiguous reductions, Cannon cyclic contraction, two-hop half contraction and grouped half row-vector contraction. The syntax registry does not imply all combinations of attributes are meaningful or implemented. The corresponding IR verifier must accept the complete strategy.

The grouped strategy uses `broadcast=host_rows`, `reduce=grouped_two_tree`, `groups=<unsigned>` and `result=replicated_columns`. The two-hop strategy uses `exchange=two_hop`, `initial_align=bidirectional`, `reduce=local` and `overlap=double_buffer`. Both explicitly request `compute=dsr fp=relaxed` and typed half tensors. Their execution and precision scopes are documented separately.

One annotation per statement is supported. Blank lines may intervene; pragma stacking, multiline pragma continuations, macro expansion and arbitrary C++ helpers remain outside the current frontend. Unsupported syntax is rejected rather than silently ignored even though native Clang runs with unknown-pragmas warnings disabled.

## Adding a capability

A syntax addition alone does not implement a feature. Development requires:

1. A declarative attribute schema and negative syntax tests.
2. Explicit typed semantics and rejection of unsupported combinations.
3. A spatial plan defining ownership, numerical order and resource lifetimes.
4. Generated CSL or a CSL library interface that realizes those semantics.
5. Native and independent numerical checks, then actual SDK compilation/execution and protocol inspection.
6. A matched, scoped performance comparison when qualifying the numerical profile.

Buffer overlap describes the runtime protocol to generate; it does not promise a hardware initiation interval. `fp=relaxed` permits documented floating-point reordering and retains explicit numerical acceptance checks. Diagnostic sampling is separately selected by the compiler's supported `--instrumentation` modes.

## Native SDK pencil FFT

`rows=P cols=P partition=pencils exchange=sdk_transpose compute=sdk_fft result=input_layout fp=relaxed` selects the explicit cubic C2C library lowering. Dimension, direction and normalization belong to the typed `spatial::fft3d` call; they are numerical semantics, not placement attributes. See [distributed FFT](DISTRIBUTED-FFT.md) for input layout, control-wavelet ownership, optional seven-stage endpoint observations and measured qualification scope.

The alternative `result=transposed_pencils` retains the post-transform ownership and omits the final two transposes. Typed planning/code generation and the32³/8x8 six-call SDK path are qualified, with five-stage endpoint audit and exact logical output-bit comparison. Output layout is an explicit contract, not an implicit change to the mathematical FFT.

## Distributed row normalization

```cpp
#pragma csl dataflow rows=8 cols=8 partition=tiles reduce=bidirectional_chain weights=feature_columns accumulation=f16 math=sdk_half compute=dsr fp=relaxed
 auto normalized=spatial::rmsnorm(x,w,0.000001);
```

`rmsnorm` defines standard per-row RMS normalization, with per-feature weights and a positive literal epsilon. The pragma explicitly selects half accumulation and the SDK half math implementation. Local matrix columns are contiguous row vectors; the reduction crosses PE columns and broadcasts row statistics. Weight ownership and normalization-factor ownership are different axes. See [distributed RMSNorm](DISTRIBUTED-RMSNORM.md) for source corrections, exact target-bit checks, half accuracy limits and measured scope. Mixed/wider precision and arbitrary composition are not implemented by this policy.

`softmax` uses `partition=tiles reduce=max_sum accumulation=f16 math=sdk_half compute=dsr fp=relaxed`. Optional `elementwise=map` requests the measured CSL `@map` exponent lowering; omission retains the existing scalar loop. Shape and scale belong to the typed call, while the pragma controls implementation. Both modes retain the same target-bit and standard-probability checks.

Gated half activation uses `elementwise=map math=sdk_half compute=dsr partition=tiles` on both SiLU and multiply, with explicit rows/cols and relaxed FP. See [Gated activation](GATED-ACTIVATION.md) for the input bound and absolute subnormal accuracy term.

Pair rotation uses `partition=tiles coefficients=per_token|feature_pairs compute=dsd fp=relaxed` with rows/cols. Coefficient shape must match this ownership policy; adjacent feature pairs cannot cross a tile boundary. Input pair order is an explicit C++ template argument. See [Pair rotation](PAIR-ROTATION.md).

## Source-derived attention contractions and phase composition

QKᵀ uses a logical `transpose(k)` view and matmul with `exchange=vertical_two_hop reduce=rotating_root order=east_first overlap=double_buffer compute=dsr fp=relaxed`, plus explicit square rows/cols. This is a vertical token-block rotation with horizontal feature reduction, not ordinary matrix exchange.64×128/128×256 are qualified, as are their resident map-softmax compositions. See [score dataflow](ATTENTION-SCORE-DATAFLOW.md).

Logical column-major A:M×M times B:M×N uses `exchange=two_hop initial_align=both_axes reduce=local overlap=double_buffer compute=dsr fp=relaxed`. Both alignments occur on device; the RHS uses a strided DSD.64×128 and128×256 are qualified; evidence is tracked in [device-aligned matmul](DEVICE-ALIGNED-MATMUL.md).

The eight-node `input Q,K,V -> transpose K -> matmul -> softmax -> matmul -> output` composition reuses those policies and requires map softmax in the same region. There is no attention-specific arithmetic opcode. The64×128 sampled and128×256 counter profiles are SDK-qualified. The planner preserves phase-local ownership of buffers, DSD stride resets, queues/tasks and microthreads; the compiler does not insert intermediate SDK transfers. The initial contract is supplied Q/K/V, unmasked single-head attention, not complete prefill/decode or a general graph fusion facility.

The batch-major normalized projection family also supports resident input rows,
feature-column outputs replicated over rows, and `fusion=collective`. See
[Batched projection fanout](BATCHED-PROJECTION-FANOUT.md) for its exact RMS producer,
shape, resource, precision, and completion restrictions. A recognized directive
does not imply all branch widths or arbitrary region compositions are supported.

## Batch-major SDK-axis composition

The Decode-oriented family keeps every batch row in each local feature shard.
`axis=x|y` names the PE coordinate of the contracted/reduced dimension; it is
independent of a C++ tensor's logical row/column index. Explicit result and
replica policies make the next consumer's ownership checkable.

```cpp
#pragma csl dataflow rows=8 cols=8 broadcast=resident_rows axis=y reduce=sdk_axis result=feature_columns replicas=rows fusion=collective accumulation=f16 collective=f32 compute=dsr fp=relaxed
 auto query=spatial::matmul_blocked<spatial::f16,spatial::scalar>(normalized,wq,4);
```

Here `spatial::scalar` is the public alias for **float**, the block-merge type;
it does not select a scalar CSL contraction. `compute=dsr` selects local half
DSR/FMA computation. The literal4 belongs to numerical block accumulation:
each half partial is merged in f32 and narrowed before the explicitly declared
SDK f32 collective. It is not a promised hardware initiation interval. The native
reference merges whole-row blocks; the device first merges local blocks and
then performs the planned collective. Relaxed floating-point reordering requires
both paths to pass their independent mathematical gates.

For the implemented normalized-QKV/cache graph, these ownership transitions are
explicit:

| Stage | Producer ownership | Reduction | Result ownership |
| --- | --- | --- | --- |
| RMS | Features on Y, replicated on X | SDK Y SUM | Same features on Y |
| Q/K/V projections | Input features on Y, weight output features on X | One fused SDK Y SUM | Features on X, replicated on Y |
| Q/K adjacent pairs | Features and pair coefficients on X | Local only | Features on X |
| Q × old K cache | Features on X, sequence on Y | SDK X SUM | Sequence on Y, replicated on X |
| Softmax | Sequence on Y | SDK Y MAX then SUM | Same sequence on Y |
| Probability × old V cache | Sequence on Y, output features on X | SDK Y SUM | Features on X, replicated on Y |
| Output projection | Input features on X, output features on Y | SDK X SUM | Features on Y, replicated on X |
| Residual add | Original X and output delta on Y | Local only | Features on Y |

Batch-major pairs use `partition=features axis=x layout=batch_major
coefficients=feature_pairs compute=dsr fp=relaxed`, with explicit rows/cols and
`pair_order::odd_even`. The order declares the actual four-product transform;
the compiler does not generate position angles. A pair cannot cross the local
feature boundary. Batch-major softmax uses `partition=sequence axis=y
layout=batch_major reduce=max_sum provider=sdk_axis accumulation=f16
collective=f32 math=sdk_half compute=dsr fp=relaxed`.

These attributes are composable only along supported typed graph structures and
resource plans. The current25-node graph requires three distinct output names
for residual, new rotated K and new projected V. Neither cache append nor head,
mask, dynamic shape or arbitrary region fusion follows from accepting this
syntax. See [projected-cache contract](PROJECTED-CACHE-ATTENTION.md) for precise
bounds, source correspondence and actual verification status.

### Explicit RMS statistic in the resident attention/FFN composition

`statistic=mean` extends the batch-major RMS policy with the same `axis=y`,
`accumulation=f16`, `collective=f32` and `math=sdk_half` attributes. It represents
the reduced statistic as a mean: widen the local half square sum, divide by the
declared global feature count in f32, reduce/broadcast through the SDK, then
narrow. The RMS consumer must not divide by that count again. The feature count
and divisor are part of the typed edge and must agree; the implemented helper
requires a power-of-two divisor. The parent proves the local sum remains finite.
This is not a general overflow cure for an already-overflowing local sum.

The supported 35-node composition consumes the actual attention residual and
shared gamma, maps UP/GATE to Y-reduced/X-owned hidden features, applies local
SiLU/product, and maps DOWN through X reduction back to Y-owned features. The
final add consumes the original attention residual. QKV and UP/GATE each fuse
their compatible reductions; the whole graph has one device launch/completion.
The public three-output contract remains final result, new rotated K and new V.
See [mean-statistic implementation and evidence](MEAN-STATISTIC-RMS.md). Acceptance
of this syntax is separate from completion of a particular SDK qualification.
