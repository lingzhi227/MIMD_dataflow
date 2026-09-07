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
