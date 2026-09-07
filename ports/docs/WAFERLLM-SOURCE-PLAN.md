# WaferLLM numerical schedule prerequisites

Current checkpoint: two-hop MeshGEMM64/128/256 and grouped MeshGEMV128/512 are separately SDK-qualified with warm calls and source-native controls. See TWOHOP-HALF-CONTRACT.md and GROUPED-HALF-GEMV.md. The original prerequisite notes below remain historical; full Prefill/Decode composition is still open.

Pinned source commitfd1c2daae37cd68706c03fc8009887ecee9900f8. This plan follows
Cannon in the numerical queue and is not a completed port.

MeshGEMM and MeshGEMV both use f16 input, output and accumulation operations.
A faithful port therefore needs typed f16 frontend/native semantics, exact
16-bit SDK transport, a target-tested FMA/rounding contract and precision-aware
error bounds. Replacing these kernels with f32 SUMMA/GEMV would neither preserve
the algorithm nor provide a meaningful performance comparison.

MeshGEMM uses a twelve-color two-hop topology, two buffers per operand,
bidirectional initial X shifts and a host W block permutation. Its host
`assignId` recurrence defines a nontrivial block-row permutation; ordinary
row-major input tiling is insufficient. X tiles are column-major, W tiles
row-major and output tiles column-major. The communication library transports
packed half values with SIMD-max fabric DSDs and requires even tile extents.

The communication module owns queues2/3/4/5 in both banks, microthreads1–4,
DSRs2–7 and a local completion task22. The compute body owns DSR1 and uses
`@load_to_dsr`, saved address progression, `@map` and `@fmach`. Tasks23–26 join
X/Y completion and computation. Buffer and DSR lifetime must be modeled across
communication/computation overlap; the same buffer cannot be reused early.

The first qualification should reload changed X/W inputs for each independent
invocation and reinitialize every pointer, shift counter and task gate. The
upstream performance launcher validates the first result, then performs timed
repeats without validating each repeated result. Its repeat loop must not be
assumed numerically correct for a new warm-call API merely because it returns.
Round-by-round block ownership and result witnesses should be independent of
upstream's permissive final relative-error screen.

Before committing f16 semantics, use actual SDK2.10.1 arithmetic/transport probes
to distinguish fused from split rounding, normal/subnormal behavior, halfway
rounding and f16 accumulation from f32 accumulation. Keep exact-zero and signed
cancellation fixtures alongside scaled random matrices and original-source
comparisons. Report numerical equivalence separately from any fixed accuracy
screen or inference-level quality claim.

MeshGEMV additionally uses PE groups and two-phase root reduction for a row
vector times a matrix; it is not the existing A*x broadcast-column profile.
Its group partition, roots, result ownership and communication accumulation
order must be recovered from its own source and host helpers.


## Implemented qualification checkpoint (2026-09-06 17:44 UTC)

The arithmetic probe and first typed two-hop MeshGEMM lowering now exist. See
[TWOHOP-HALF-CONTRACT.md](TWOHOP-HALF-CONTRACT.md) for the declared binary16 order,
SDK acceptance, exact internal checks and remaining scale/performance work.
The earlier prerequisites above are retained as source reasoning, not a claim
that current implementation is still equation-only.

The next source-distinct MeshGEMV lowering must describe a row-vector contraction
with a group reduction policy, not reinterpret an ordinary column-vector GEMV.
Its pinned host replicates each X row segment across PE columns; W tiles are
row-major. Source `@map(gemv_static_step, X_dsd)` executes DSR1 half FMA across
each local matrix row. Communication reserves queues 2–7 in both banks and
DSR source1 slot 2. Two phase-specific group roots feed an allreduce followed
by a broadcast, so the SDK must test every replicated output, changed warm
calls and both group boundaries, not merely collect one root's output.

Local half FMA and half additions in the reduction need separate prefix
observations: the independent oracle must reproduce the actual tree addition
order from source roots, and retain an original-product error check. Exposing
the group partition and result replication in the typed schedule will make
those ownership and queue lifetimes reviewable before CSL generation. No
MeshGEMV SDK qualification is claimed by MeshGEMM evidence.
