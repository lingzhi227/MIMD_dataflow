# Resident batch-major FFN — qualified bounded configuration

`hls.cpp` expresses RMS normalization, fused UP/GATE projections, stable SiLU,
pointwise gating, DOWN and residual. B=5, model features256, hidden features512,
and8×8 PEs. The shape follows the previously qualified UP/GATE configuration.

Input/model features are sharded over Y and replicated over X. Hidden features
are sharded over X and replicated over Y. Weights remain resident. Local
arithmetic is f16; independent SDK X/Y reduce+broadcast operations accumulate in
f32 and narrow once. This is an explicit precision policy, distinct from the
source Decode grouped-half collective. SDK half stable SiLU replaces the source
fast_exp approximation and retains an explicit approximation/underflow limit.

Source: WaferLLM Decode commit fd1c2daae37cd68706c03fc8009887ecee9900f8,
`Decode/src/decode.csl` RMS, vecmat_computation, UP/GATE fusion and gated DOWN
residual. Local DSR kernels retain attribution in their reusable CSL modules.
This is not full Decode, cache management or hardware performance qualification.

Every fresh run preserves parsed/typed IR, schedule and memory plans, native
binary/results, generated CSL and frozen implementation. Native observations
check individual mathematical stages, particularly DOWN before residual addition.
Eight-call SDK qualification and the matched local-compute control have passed; see `docs/BATCHED-FEED-FORWARD.md` and the PORT.json qualification link.

Completed evidence: SDK run200435 passes eight calls, with linked static footprint
38400B/PE. Source-compute control201155 retains
original Decode gemv_static_step/vecmat_computation under the same SDK schedule;
this isolates local lowering overhead. Native202543 adds shared runner gates and
debug/partial-audit support, emits identical CSL, and has no additional SDK run.
`experiments/build_batched_ffn.py` provides a fresh-build entry point; the unified project runner now supports the registered fixture.
The debugger steps0..9 expose local/reduced RMS, normalized input, local/reduced
UP/GATE, activation, product, local/reduced DOWN, and final residual respectively.
