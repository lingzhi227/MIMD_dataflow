# Cannon cyclic block matrix multiplication

The frontend uses the existing typed `spatial::matmul` with a distinct policy:

```cpp
#pragma csl dataflow rows=4 cols=4 exchange=cyclic initial_align=host reduce=local fp=relaxed compute=vector
```

This preserves the Matrix-algorithms Cannon dataflow, rather than selecting
SUMMA broadcasts. Host transfer performs only initial block permutation;
PE(x,y) receives A[y,(x+y)%P] and B[(x+y)%P,x]. Round r accumulates block
k=(x+y+r)%P, then A shifts left and B shifts up. Each axis has wrap and two
alternating neighbor colors. Even coordinates send first, odd coordinates
receive first. The even mesh and sequential A/B exchanges avoid cyclic waits
and reuse one temporary matrix tile. The vector body retains row DSD FMA.

The initial accepted plan supports square divisible matrices on4×4 or8×8
meshes, subject to a48KiB per-PE estimate and signed DSD extent bounds. It
rejects unsupported geometry rather than silently using a different algorithm.
Layouts use queues2/3, reserving0/1 for SDK memcpy. Logical PE coordinates are
explicit parameters. Dedicated host input pointers remain fixed while working pointers rotate.
Every external invocation resets C, working pointers and DSD bases; rotated buffer ownership from a previous invocation cannot leak into
the next one. The old source `comptime_struct` parameter and RPC launch syntax
are migrated to SDK2.10.1. The first failed SDK compilation is retained.

## Evidence and debugging

Each bundle includes frontend source, checked/optimized IR, spatial plan,
generated CSL, implementation snapshot, native executable/reference and real
SDK results. The shared matrix SDK binding performs typed I/O and initial
permutation; all contraction and ring movement execute in CSL. A bundle can
be executed through `experiments/execute_frozen_bundle.py` without modifying
another running batch's canonical toolchain.

Auditing regenerates schedule and CSL, checks every cumulative tile against
original inputs in cyclic K order, and checks the first/last elements of both
received blocks at every round. Final results must exactly match the last
history tile. Shift/round/invocation counts, drained queue masks and timestamps
are separate protocol checks. A componentwise f32 dot-product error bound is
the arithmetic contract; the fixed3e-5/3e-6 accuracy screen is also reported.
Zero-input and halfway-rounding fixtures give exact reset/rounding witnesses.
The debugger reports global K indices, tile memory order, expected/actual
partial result, block witnesses and lifecycle state for a selected PE/round.

64×64 vector run162846797553 passes four SDK invocations and65,536 intermediate
values, maximum absolute error2.53283e-6; the fixed screen passes. Static SRAM
is14,224/49,152 bytes, not runtime stack usage. Scalar run163022039496 also
passes all four calls. Twelve coherent numerical/protocol mutations in
`evidence/cannon-audit-mutations-20260906T163717300211Z.json` are rejected.

`evidence/cannon64-vector-scalar-comparison.json` reports7.114× local compute,
6.763× maximum-local resident interval, and2.399× total simulator cycles for
vector versus scalar on the same four inputs and schedule. The total includes
host transfers and diagnostics; the local resident interval excludes transfers.
These are simulator comparisons, not hardware speedups or vendor-BLAS claims.

`evidence/cannon-native-20260906T163614178201Z/comparison.json` runs the pinned
original compute/ring source with explicit SDK ABI migration, warm reset and
timestamps on two inputs. HLS/native outputs are bit identical. HLS histories,
counters and witnesses add1.1365% maximum-local interval overhead. The native
adapter reuses the HLS layout ABI and pure host permutation; it is not an
unmodified upstream binary. All transformations and source hashes are retained.

128×128/64PE run163023774095 completed four device calls but failed audit on
calls2–4. The first call has correct block witnesses and a6.604e-6 maximum
absolute final error; subsequent calls use wrong blocks. SDK memcpy follows
the exported pointer cell at transfer time. Exporting the rotating working
A/B pointers therefore redirects the next host upload before `main` resets
those pointers. Three-buffer rotation has period3: P4's three shifts hide the
bug; P8's seven shifts expose it. The fix exports dedicated invariant handles
for Matrix_1 and Matrix_2 and retains separate working pointers. The native
adapter has the same ABI correction. A targeted generated-pointer ownership
regression demonstrates both the P4 masking case and P8 failure of the old
exports. Fresh128run164836013765,64vector165014750658 and64scalar165016859494
pass all four calls; the original failure and inputs are preserved unchanged. This profile does not yet cover odd
meshes, arbitrary rectangular products, incomplete tiles, asynchronous overlap,
larger-scale hardware execution or a lean production instrumentation mode.


## Corrected qualification

The fixed128profile audits524,288 intermediate values, maximum absolute error
6.604013e-6; all fixed accuracy checks pass. Identity and zero-input epochs are
exact. Static footprint remains18,432/49,152 bytes (not stack high-water).
`evidence/cannon128-host-binding-fix-comparison.json` proves identical source,
inputs, semantic IR, schedule and native reference between failure and fix.
The two-call corrected native128adapter in
`evidence/cannon-native-20260906T165940999152Z/comparison.json` passes, with
HLS/native maximum-local ratio1.0112438. Corrected native64control165604674620
retains ratio1.0113652. The corrected vector/scalar comparison is
`evidence/cannon64-fixed-vector-scalar-comparison.json`; the earlier comparison
and failed large execution remain available. Twelve mutations against the
corrected64bundle reject in cannon-audit-mutations-20260906T170138634345Z.

These three bounded profiles enter catalog73. Shared SUMMA transport regression
163304473241 passes4SDKcalls;79unit tests pass after the ABI regression. Full
source-project coverage and the remaining numerical queue are not complete.
