# Resident real BiCGStab

`spatial::bicgstab_csc<N,MaxIterations>` is the unpreconditioned real-f32
BiCGStab recurrence over a canonical CSC operator. Initial lowering is512
unknowns,4608 stored entries,4x4 PEs, and at most32 completed updates. It uses
the common resident solver services for transpose, SDK train SpMV, scalar
collectives, stable norms, queue ownership, I/O and final completion. The
algorithm-specific controller contains the shadow residual, p/v/s recurrence,
stabilization and branch witnesses. Ten-case SDK153005874768 qualification passes and it is a bounded catalog entry.

The pinned SDK source is `benchmarks/bicgstab/src/kernel_bicgstab.csl`. As in
that source, the initial implementation performs separate t dot s and t dot t
reductions. Packing those reductions together is a possible measured optimization,
not an assumed feature of the present implementation. This general-CSC schedule
is distinct from the source's seven-point stencil operator and synchronized
reference-clock timing; neither schedule nor performance parity is claimed.

An early-s convergence check follows conventional BiCGStab practice, visible in
[SciPy's implementation](https://github.com/scipy/scipy/blob/main/scipy/sparse/linalg/_isolve/iterative.py).
It commits x+alpha*p, records the completed update, and skips the t/omega stage.
This handles an exact scaled-identity solve without dividing0/0. The requested
residual threshold is max(rtol*norm(b),atol), consistent with the
[SciPy interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.bicgstab.html).
The project uses its own result ABI and exact-zero/nonfinite guards; it is not
an implementation of SciPy's epsilon-based breakdown thresholds.

A zero/nonfinite shadow denominator or unusable omega returns numerical
breakdown. Failed partial steps leave the last completed solution unchanged.
The completed-iteration count includes an early-s step. Squared-norm underflow
is distinguished from a zero vector through a scaled norm. The final result
always recomputes b-Ax. Negative shadow products are allowed; the CG-specific
nonpositive-curvature reason is not used here.

The public history records the unweighted residual of completed iterates.
Diagnostics separately record rho, shadow dot v, s dot s, t dot s, t dot t,
alpha/omega/beta, early-s selection, failure stage, and individual operation
counts. An audit replays the exported trajectory against original CSC entries,
checks branch-dependent SpMV/collective counts and quiescent queues, and reports
native iteration/solution-screen differences separately from true residual
acceptance. Other nonfinite breakdown trajectories remain unqualified.

Acceptance fixtures include nonsymmetric row-diagonally-dominant systems, zero
RHS, exact initial solution, budgets0/1, zero operator, a changed nonzero initial
solution, tiny squared-norm underflow, scaled-identity early-s completion, and
blocks[[1,1],[1,0]] with RHS[1,0] giving zero t dot s but positive t dot t.
The last fixture exercises zero omega before any solution is committed.

The shared callback layer has passed two full CG and two PCG regression calls
in151932991873 and151933471926. BiCGStab153005874768 links at47584/49152 static
bytes, leaving1568; this is not runtime stack headroom. Legacy internal ABI names
`mesh_cg.v1`, `cg_*` and `f_cg` are currently shared by these solver variants;
the typed operation selects the algorithm. Debugging labels distinguish BiCGStab
omega from CG curvature and show failed attempts separately from completed steps.

## Accepted evidence

All ten SDK calls pass, including strict original-system residual checks for
every reason0. The two nonsymmetric systems take7 and6 updates and both select
an early-s final update. Zero operator reports stage21; zero omega reports
stage25; both keep the initial solution unchanged. The scaled identity completes
in one early-s update with exact zero residual. Full68-profile CPU153331759931
and73 unit tests passed before promotion.

- SDK: `run-20260906T153005874768Z`, static footprint47584 bytes.
- Selected diagnostic-store control: `evidence/bicgstab-diagnostic-control-20260906T154035622979Z/comparison.json`; ratios1.0048164 and1.0057696 for two full solves. Stores used as live continuation state remain; this is not the cost of all instrumentation or original SDK performance parity.
- Fourteen coherent/transport/branch mutations rejected: `evidence/cg-audit-mutations-20260906T154438011315Z.json`. The report points to a replayable auditor/toolchain snapshot.
- Zero-omega debugger view: `evidence/bicgstab512-debug-zero-omega.json`.
