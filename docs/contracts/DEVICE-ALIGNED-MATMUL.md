# Device-aligned logical-layout matrix multiplication

The HLS algorithm is ordinary `matmul(a,b)` with policy:

```cpp
#pragma csl dataflow rows=8 cols=8 exchange=two_hop initial_align=both_axes reduce=local overlap=double_buffer compute=dsr fp=relaxed
auto product = spatial::matmul(a,b);
```

`mesh_device_matmul.v1` matches typed dataflow and policy, not application names. The initial bounded family is A:M×M and B:M×N, half precision, P×P PEs, P4 or8, divisible tiles and supported packed lengths. Host inputs are immutable logical column-major tiles. Resource/memory/offset checks reject unsupported configurations before CSL compilation.

The pinned MeshInfra/WaferLLM `output_matmul` expects a prealigned row-major RHS. Directly feeding column-major prior-stage V tiles gives relative L2 errors around1.414/1.454 in the isolated source test. The diagnostic host prealignment proves the mismatch but is not a resident solution. The explicit device adapter aligns RHS vertically by column through existing two-hop communication, then aligns LHS horizontally by row. Local RHS DSD strides by tile height; contraction advances by one half element. No local transpose allocation or host permutation is required. Existing asynchronous communication overlaps DSR FMA; completion tasks join both directions before swapping buffer roles.

Colors1..11 are initialized through the shared source-derived router library. Queues3..7 are owned; UT0..3 support both axes. Tasks19/20/25/26 coordinate completion. Compute DSR1 and communication DSR3/4 have distinct lifetimes. This is not an unlimited resource abstraction. The SDK memcpy channel/halo and Python runtime transport remain explicit in every frozen bundle.

64×64 times64×128/P8 has six sampled and six counter SDK calls. Fixtures include identity, normalized random probability, zero and changed warm inputs, uniform/rolled probability and independent signed dense operands. All sampled accumulation prefixes and both live operand owners match half source trajectories exactly. Independent actual C++ and device checks use original-input math.fsum, with .015 relative L2 and .02 peak-scaled bounds; per-component relative accuracy is not implied.

Registration: `evidence/qualification-20260907T030744834095Z.json`.36 sampled and15 counter coherent corruptions are rejected. Linked static maximum13328B leaves35824B before runtime stack usage. Counter comparison9853/9717 source cycles (~1.40% overhead) includes immutable input copies; sampled10613/9993 (~6.20%) also includes live operand observations. The control is the explicitly adapted source, not unmodified upstream. Measurements are local WSE3 simulator intervals, not hardware throughput.

The128×128 times128×256/P8 sampled bundle025356540275 now passes six SDK calls,36 mutations,3,932,160 internal half observations and matched source controls; qualification035515867251. Its linked static maximum is32928B. The distinct counter034139953919 passes six calls and15 mutations:35006/34799 source cycles (~0.595% overhead). Sampled36726/35462 (~3.56%) includes extra live-operand observations. Complete resident attention, masking, heads, KV cache and full prefill/decode remain separate work. Source adapters belong to experiments; numerical oracles and source-authoring scripts are not part of compiler/codegen.
