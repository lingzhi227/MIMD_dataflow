# Resident CG contract under implementation

The typed frontend, C++ reference, interpreter and bounded distributed CSL
lowering are implemented for512 dimensions/4x4 PEs and at most64 iterations.
Eight-case SDK execution and independent supplemental audit pass. Complete code links after moving
large inline primitive paths to single dispatcher sites; the current linked
static high-water is45376/49152 bytes. This is not a stack high-water bound.
`mesh_cg_512_4x4` is in the catalog with bounded numerical and performance scope.
The initial full audit failed on the deliberately underflowing squared residual;
the original execution/failure is preserved, and the narrowly corrected audit
resides in a separate evidence copy.

The initial result is a record with `solution:f32[N]`, `reason:u32`,
`iterations:u32`, `residual_squared:f32[MaxIterations+1]`, and
`true_residual_norm:f32`. Callers explicitly export its fields. Status and
iteration counts do not travel through a floating-point representation.
The runtime iteration limit is u32 and may be zero; it cannot exceed the static
history capacity. Tolerances contain relative then absolute values.

The convergence threshold is max(relative_tolerance * norm(b),
absolute_tolerance), following the familiar unpreconditioned residual criterion
in [SciPy CG](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html).
Distinct termination reasons follow the practice of reporting numerical failure
separately from convergence and iteration limits, as exemplified by
[PETSc convergence reasons](https://petsc.org/release/manualpages/KSP/KSPConvergedReason/).
The enum values are project-specific; this is not SciPy/PETSc ABI compatibility.

| Reason | Meaning |
| --- | --- |
|0 converged | True residual check meets the threshold |
|1 iteration_limit | Budget exhausted without certified convergence |
|2 nonpositive_curvature | CG encountered nonpositive p dot A p |
|3 numerical_breakdown | Nonfinite arithmetic or nonzero residual with an unrepresentable squared norm |
|4 residual_gap | Recursive residual appeared converged but recomputed true residual did not |

CG assumes an SPD operator. This is not an eigenvalue/symmetry certification
algorithm, and an indefinite matrix need not always trigger reason2. The native
reference uses canonical CSC SpMV, f32 reductions, FMA vector updates and a stable
norm for initial/true-residual decisions. It detects squared-norm underflow both
initially and after an update, and checks beta before updating the direction.
No global floating-point reassociation equivalence is claimed. Near a tolerance
boundary, different valid reduction orders may change iteration counts; the
audit uses explicit margins and the original-system residual rather
than demand unconditional exact iteration agreement.

`residual_squared[0]` is the initial recursive square norm. Entries1..iterations
record completed updates; the unused tail is zero. The true residual is computed
separately from b-Ax and does not silently overwrite the recurrence history.
Nonfinite failures may retain nonfinite numerical diagnostics; a broader SDK
serialization/audit must represent them explicitly instead of accepting them
under ordinary numerical tolerance checks.

Original sources inspected are pinned SDK `conjugate-gradient/src/kernel_cg.csl`
and `blas.csl`. Their recurrence uses memory DSDs, `@fmacs`, local dots and global
reductions. This implementation targets general CSC and transpose ownership,
not the source's specific seven-point stencil layout. Device vectors, recurrence,
iteration limit and termination must stay resident. No host numerical iteration
is an acceptable lowering of the resident pragma.

The accepted cases cover representative SPD sparse systems, changed-input warm
calls, zero RHS, an exact supplied solution, limits0/1, controlled nonpositive
curvature and initial square-norm underflow. The canonical catalog run
`run-20260906T145225072459Z` passes all eight with the corrected audit.

## Resident schedule and qualification

The frontend explicitly selects CSC train exchange, row/column scalar reductions,
transpose redistribution, vector DSD computation, relaxed floating order and a
resident recurrence. Each invocation transports the original packed sparse
operator, RHS, initial solution and controls once; Python launches `f_cg` once.
No host numerical iteration supplies device values. The CSL controller owns
transpose, SpMV, dot products, FMA updates, stable norms and termination.

The dispatcher shares the existing collective callback task10. Primitive requests
store continuation state and activate that task, with one call site for each
large primitive. Sparse completion schedules the same continuation. This avoids
large inline callback duplication without allocating another local task.
Communication boundaries check drained application queues and explicitly restore
SDK queue colors and filter enables, using the unchanged SDK collective library.

Audit reports distinguish the strict original-double-operator condition
`norm(b-Ax) <= requested_threshold` from the declared rounding allowance
`0.05*requested_threshold + 1e-7*norm(b)`. Passing the latter must not be described
as satisfying the former. Device reason0 refers to the device f32 recomputation.
The acceptance fixtures have explicit expected reasons; agreement with native
iteration counts and the fixed cross-solution screen is reported separately.
Both the native and device original-system residuals must pass for converged
qualification cases; device replay uses its exported alpha/beta trajectory.
The current finite diagnostic audit does not qualify arbitrary nonfinite
intermediate trajectories, or their additional failed-SpMV callback counts.

## Evidence

- Full SDK execution: `projects/sdk_examples/mesh_cg_512_4x4/run-20260906T143749385026Z`.
- Corrected independent audit: `evidence/cg-audit-20260906T144514422211Z/provenance.json`.
- Nine injected faulty records rejected: `evidence/cg-audit-mutations-20260906T144826863907Z.json`.
- Instrumentation control: `evidence/cg-diagnostic-control-20260906T144727279006Z/comparison.json`,
  two complete solves, maximum-local ratios1.008312 and1.009709. Same primitive
  schedule with diagnostic stores removed; protocol checks/result history remain.
  This is not original SDK full-CG performance parity or hardware performance.
- Shared stable norm regression: `run-20260906T144736633986Z`, four warm calls.
- Shared SpMV regression: `run-20260906T144928356650Z`, four warm calls.

The two nontrivial SPD solves took9 and8 updates; original-system residuals
0.00062475655 and0.00129109397 are below strict requested thresholds
0.00133569713 and0.00132214584. Zero RHS and exact initial solution take0
updates. Limits0/1, negative curvature and initial square-norm underflow return
the intended distinct reason. Maximum local measured cycles per case are
788234,145508,145183,212929,207087,150355,669937,148156.

A matched diagonal control exposed a real stopping-order difference:
`run-20260906T150210286027Z` takes12 device updates versus13 native updates.
The fixed native/device solution screen(3e-5 relative,3e-6 absolute) fails57/512
components, with maximum difference0.00010058. Both original-system residuals
pass the requested threshold:0.00023849013 device,0.00132074346 native, versus
0.00226274164. Neither iteration identity nor agreement between two differently
rounded approximate iterates is a universal solver correctness criterion.
These failed checks remain recorded; the successful separate residual-contract
audit is `evidence/cg-diagonal-audit-20260906T151210230052Z`. It still checks
device trajectory replay, original operator, scalar identities, protocol state
and each result's original residual. It reports the fixed screen as false.
