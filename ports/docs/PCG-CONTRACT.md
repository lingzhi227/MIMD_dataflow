# Resident Jacobi-PCG

`spatial::pcg_csc<N,MaxIterations>` uses the same structured solver result and
explicit spatial dataflow as `cg_csc`. It adds a Jacobi preconditioner to the
resident recurrence, following pinned SDK
`benchmarks/preconditioned-conjugate-gradient/src/kernel_pcg.csl`.
The current bounded lowering is512 dimensions,4608 stored entries,4x4 PEs and
64 maximum updates. Nine-case SDK qualification145939542536 passes and this profile is in the
bounded catalog. Complete linked static SRAM
high-water is46224/49152 bytes (2928 remaining), not runtime stack evidence.

The original canonical CSC diagonal must be present, positive, and at least
2^-16; all ordinary finite input bounds still apply. The frontend/native
reference and packing validation reject invalid diagonals. This bounded input
contract is not an SPD certificate. A positive-diagonal indefinite test explicitly
checks that curvature failure remains distinguishable from convergence.

Packing copies original diagonal entries into the solution vector ownership;
it performs no reciprocal or numerical iteration. CSL computes the inverse
locally, applies it with a memory DSD multiply, and uses r dot z for alpha/beta.
The extra diagonal and weighted-inner-product witnesses are audited against
original entries. The input matrix itself still travels through shared CSC train
SpMV and the resident vector redistribution uses the same SDK collectives.

The public residual history remains r dot r. It is never replaced by r dot z.
The initial and final residual norms are scaled norms, and successful termination
requires a device recomputation of b-Ax. Independent audit separately reports
whether the original double-accumulated system meets the strict requested
threshold, with any rounding allowance visible. The supplied initial solution,
zero/one iteration budget, nonpositive curvature, tiny squared-norm underflow and
warm-call reset behavior are part of qualification.

Nine fixtures include diagonally scaled sparse SPD systems, zero RHS, exact
initial solution, budgets0/1, positive-diagonal indefinite blocks, a changed
nonzero initial solution, tiny input, and a diagonal system with nine distinct
powers-of-two coefficients. For the latter, the native reference takes13 updates
without preconditioning and1 with Jacobi. The matched-input CSL comparison uses12 CG device updates and1 PCG update.
Maximum-local intervals are942350 and226939 cycles, a4.15244 ratio for this
diagonal input only. It includes device inverse work, excludes H2D/D2H, and
compares two warm CG calls with one warm PCG call. This is algorithm-level
preconditioning benefit, not a universal PCG or hardware speedup.

This is the SDK PCG recurrence applied to a general CSC operator, not a claim to
have reproduced its original seven-point stencil layout, synchronized timing
protocol or end-to-end performance. Those source-specific schedules remain in
the queue. CG and PCG share the continuation dispatcher, storage/route planning,
SDK bindings and most runtime code; Jacobi support is selected from the typed
algorithm operation, not inferred from an application filename.

## Evidence

- SDK/native/trajectory/protocol: `run-20260906T145939542536Z` (nine calls).
- Shared CG regression: `run-20260906T150025996264Z` (two full calls).
- Matched comparison: `evidence/pcg512-matched-diagonal-comparison.json`.
- Eleven coherent audit faults rejected: `evidence/cg-audit-mutations-20260906T151359000508Z.json`, including inverse and weighted-inner-product corruption.
- Canonical catalog CPU run: `run-20260906T151359077101Z`.

CG control has an explicit native/device fixed solution-screen failure and
iteration-count difference; both approximate solutions meet their strict
original-system residual threshold. See the CG contract for preserved failures
and corrected audit provenance. No failed screen is relabeled as passed.
