# Resident solver composition: resource evidence and pending work

This is a design/experiment record, not a completed solver. The next application
must keep the numerical recurrence and termination on the device. Host-driven
SpMV/dot iteration would not meet that contract.

## SDK task-slot probes

`experiments/task_slots/run.py` imports actual SDK2.10.1 memcpy, binds one local
task, then compiles and increments a device counter in three host launches within
one runtime. WSE3/default context/nonstreaming memcpy is the tested scope.

- `task-slots-20260906T132115614149Z`: slots8,24,26 compile and produce[1,2,3].
  Slots0,7,21,27,28,29,30,31 are compiler-rejected; exact diagnostics retained.
- `task-slots-20260906T132158705486Z`: slots22,23 also compile and execute[1,2,3].
  Routable color22/23 reservations do not imply a same-number local task
  reservation on this target. Keep resource namespaces separate.
- First attempt132022914059 stopped at a simulator-construction assertion after
  a previous runtime instance had stopped. Isolated worker processes resolved
  this harness limitation; it is not evidence that local slot24 is invalid.

SDK `memcpy/sys_params.csl` declares local24 for persistent-context entry.
Successful default-context execution does **not** authorize reusing24 in a
persistent-context program. The installed memcpy implementation binds local21,
27,28,30; compiler rejection also protects special29/31. Avoid deriving resource
availability from a comment alone, or promoting an isolated probe into a claim
about arbitrary import combinations.

## Candidate composition

The existing train SpMV binds14 local tasks. X/Y SDK collectives require four
more. A scalar-collective phase FSM uses one callback task instead of four.
Converting SpMV initialization from an asynchronously activated task into a
synchronous function can release its task10; this change passes512/4096 native and four-call SDK regression
(run-20260906T132458488734Z).
The13+4+1 candidate fits18 task slots without assigning a local task to the
solver itself: host entry and library callbacks can call controller functions.

Candidate allocation: SpMV11–20 and24–26, CCL8/9 and22/23, scalar callback10.
This is not yet validated as a combined program. The production standalone
SpMV planner still conservatively avoids22/23. Do not relax global reservations
without carrying the precise runtime/context contract into the target model.

## Corrected queue ownership: combined compiler evidence

The initial candidate incorrectly interpreted the SDK `queues=[q0,q1]` as
one input and one output queue. The actual SDK source explicitly uses both
q0/q1 in **both** banks, for its two routable colors. Two dimensions therefore
occupy four input and four output queues. The standalone reduction schedule
metadata has been corrected to input/output[2,4,3,5]. Historical snapshots keep
the incorrect ledger but their actual CSL execution is preserved unchanged.

Combined attempt `resident-composition-20260906T133803273116Z` is rejected for
duplicate input-queue initialization. The earlier133702401920 attempt had an
experimental authoring error (second layout block), also preserved. A static
13+4+1 task allocation is feasible, but the proposed disjoint queue allocation
is not. Task capacity does not establish combined-library legality.

The installed SDK `<tile_config>` provides `input_queue_config` and
`output_queue_config` helpers to reassign drained queues while preserving other
fields. The next experiment leaves SDK collectives unmodified, lets the enclosing
phase owner initialize SpMV queues, and switches queue associations only after
completion callbacks and checks that all non-memcpy queues are empty. Collective
COLOR_0 retains its SDK scatter filter association; SpMV turns filtering off.
Both transpose directions, SpMV and scalar sum must execute in one host launch.
This is under SDK validation, not yet an accepted resource-lifetime proof.

## Ownership conversion

For a square P-by-P mesh, SpMV input vector block order is column-major in PE
coordinates, while output ownership is row-major. A resident CG recurrence must
transpose ownership. Candidate SDK implementation: gather each row to its
diagonal PE, then scatter down that column from the diagonal root. This requires
N/P temporary floats on a diagonal PE and preserves the logical vector order.
Budget those buffers together with all solver vectors, sparse trains and code.

Acceptance remains a real HLS program, staged lowering, repeated SDK runs,
original-system residual and iteration-history checks, explicit breakdown and
convergence semantics, and a scoped native CSL comparison. The earlier local
CG equation profile does not satisfy this distributed application requirement.

## Accepted bounded composition and memory restriction

`resident-composition-20260906T134727430627Z` passes four512²/16PE calls.
`resident-composition-20260906T140648455668Z` passes four4096²/64PE calls with
explicit capacities593 nonzeros,364 columns and362 rows per PE. These are the
maxima of the preserved four original matrices, not generic4096 capacity bounds.
The default-capacity134924963979 and wider compact140001193248 link failures
remain preserved. The successful compact source is separate; no original matrix,
nonzero count, input vector or standalone profile is weakened or overwritten.

The compact transpose uses32-element chunks, borrowing the SpMV north-partial
buffer only while SpMV is inactive. The gather workspace needs256u32 words,
within362f32 storage slots; both transpose callbacks complete before SpMV
initialization zeros/reuses that storage. SpMV's exported partial witness is
separate from the borrowed array. All intermediate permutations, roundtrips and
original-input outputs pass. This validates the observed schedule, not arbitrary
concurrent borrowing of a library's internal buffers.

`linked-memory.json` records all20 linked ELF classes: worst static SRAM
high-water48832 bytes, only320 bytes below48KiB. It excludes configuration
registers outside SRAM and does not measure runtime stack usage. This is not a
production-safety margin or evidence that full4096 CG state fits. The solver
starts at512 while complete code/data/task/stack planning is developed.

### Callback guarantees and remaining proof boundary

The SDK collective callback is reached through its lock/teardown machinery after
the local transfer is complete. A transpose joins row gather and then column
scatter; a result PE has received its specified chunk before completion. Before
SpMV, the reassigned input6/7 queues were CCL COLOR_1 queues, which this transpose
path does not use. CCL COLOR_0 input3/5 associations remain available. SpMV
completion requires both horizontal trains finished; each train checks receive
and transmit counts, after the vertical compute/send join. The original-entry
and all-PE zero-remaining-count audit is retained.

The phase owner additionally checks input/output empty masks for all queues2–7
before each reassociation. These checks and callback counts are necessary local
evidence. They do not alone prove absence of every possible late/in-flight peer
wavelet. Routable colors and filters retain their own identities, and the next
SDK collective performs its normal network reconfiguration/teardown handshake.
No general global-barrier or race-freedom theorem is claimed from four runs.
Future schedules must preserve these transport contracts or provide additional
peer agreement; they cannot reuse queues merely because a local send returned.
