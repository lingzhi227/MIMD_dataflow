# Resident fixed-step power iteration

The frontend operation `power_csc<N, MaxIterations>` describes a resident
sequence of sparse matrix application, global stable norm, and normalization.
The dataflow policy selects CSC train exchange, row/column collective reduction,
transpose redistribution, and vector DSD scaling. Python transports the input
matrix, initial vector and requested step count once per invocation; iteration
and termination remain on the PEs.

The initial profile is N=512, 4×4 PEs, 4,608 stored CSC entries (including explicit
zeros), at most 32 steps. This is a general sparse-operator port of the fixed-step
algorithm in the SDK's `benchmarks/power-method/src/kernel_power.csl`. The
original seven-point stencil operator, synchronized clocks and complete benchmark
layout are not reproduced by this profile. The frontend is not a claim that
all possible power-method variants have been implemented.

## Numerical contract

For each requested step compute y=A*x, n=norm(y), then x=y/n. A zero requested
step count returns the original, possibly non-unit vector exactly. A zero norm
terminates before division and preserves the last completed vector. Negative
leading eigenvalues may alternate the vector sign. Norm computation uses the
SDK stable scaling primitive and row/column collectives rather than an unscaled
square sum; the qualification includes a nonzero initial vector of order1e-30.

The result has vector, reason, completed iterations, and attempted-step norms.
Reason0 means the requested number of steps completed; it does not certify
convergence, dominance, uniqueness or a useful eigenpair. Reason1 is zero norm;
reason2 is numerical breakdown. The initial qualification addresses finite
completed and zero-norm trajectories. General overflowing intermediate states
are not qualified. Unused norm slots remain zero.

## Shared implementation and audit

The resident runtime shares single-flight SpMV, transpose, stable reduction,
queue ownership and callback continuation with CG/PCG/BiCGStab. A generated
`host-abi.json` describes typed input placement, output record fields and
exported diagnostics. No numerical recurrence is executed by the SDK host
binding. The common CSL dispatcher retains one call site for large primitives
to avoid compiler expansion exceeding PE memory.

The auditor regenerates schedule, CSL, host ABI and sparse packing. It checks
all-PE replicas, output ownership, attempted/completed counts, queue drain
witnesses and timestamp words. Every attempted iteration is replayed against
original CSC entries; norm and reciprocal checks use relative tolerance with
zero absolute allowance, including tiny norms. Returned vectors also have an
independent fixed-step reference and unit norm check when at least one step
completed. Rayleigh quotient and eigen-residual are reported separately with
`dominance_certified=false`.

The final operator witness belongs to the last **pre-normalization** input,
not the returned normalized vector. For zero requested steps no operator runs;
the operator buffers may be stale from the preceding warm call. Debugger and
auditor label those witnesses inactive and still check unchanged output,
zero histories, counters, queue state and invocation lifecycle.

## Qualification status

Eight warm native and SDK2.10.1 calls pass in
`projects/sdk_examples/mesh_power_512_4x4/run-20260906T161021057910Z`.
There are76 passing unit tests. Fourteen coherent numerical/protocol mutations
in `evidence/power-audit-mutations-20260906T161704205254Z.json` are rejected;
a separate positive check accepts stale operator witnesses only for zero steps.
Static ELF allocation is43,632/49,152 bytes across16classes, leaving5,520 bytes
unallocated statically; this is not a runtime stack measurement.

`evidence/power-scalar-control-20260906T161644010448Z/comparison.json` compares
16-step and4-step full resident calls with only normalization changed to a
scalar loop. Scalar/vector maximum-local cycle ratios are1.0091638 and1.0090842.
This measures roughly0.9% total-interval benefit on these shapes; it does not
establish a large overall speedup. H2D/D2H, hardware throughput, synchronized
global latency and original seven-point benchmark parity are outside the claim.
The input redistribution and SpMV/norm work dominate these profiles.
