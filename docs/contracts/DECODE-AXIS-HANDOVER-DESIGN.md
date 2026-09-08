# Next composition boundary: changing the reduction axis

**Design notes, not an implemented or qualified runtime.** Complete normalized
QKV and UP/GATE qualification before extending the application chain.

The current lowering intentionally keeps Y routes throughout a call. UP/GATE
produces output-feature shards on X, replicated across Y. A following down
projection naturally contracts over X and produces original-feature shards on Y,
which align with the original residual input. This ownership transition is useful
and should be represented in physical IR, rather than hidden host transposes.
It also exposes the next protocol requirement: safe reuse of the communication
resources when changing their axis.

## Why local return is insufficient

A root can finish injecting a broadcast while downstream forwarding remains in
flight. Changing those routes on local return is not a global quiescence proof.
The original source changes axes between subgraphs; the new compiler must not
infer a general safe transition merely because one source run happened to work.
Local computation between collectives may provide timing slack, but no such
static latency proof is currently implemented.

A blocking host launch on every PE is a known correctness boundary. Splitting the
algorithm into host-separated phases is a useful reference implementation, but
its host latency must be measured and must not be hidden by summing only device
cycle intervals. It is not automatically a suitable production implementation.

## Resource facts from the installed SDK

The pinned SDK image is
`fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d`.
The installed `<memcpy/sys_params>` has SHA256
`e70159b986fd92a4f4a83a0a46abe209ce64ed45903ac8ccda9ca0c7a4ef47df`.
Its default core-rectangle declarations reserve colors20–23, input/output
queues0/1, local tasks24/27/28/30, and control tasks33/34/35/36/37/40.
The actual WSE3 D2H implementation additionally derives and binds local task21
from default D2H color21. The contract includes that implementation binding;
24 is a conservative persistent-context reservation.
`toolchain/sdk2101_resources.py` checks application collisions with these defaults.
This is a versioned default-memcpy contract, not a universal allocator or proof
that other imported libraries allocate nothing.

The present application uses colors5–9 and queues3–7. A future control plane
could investigate queue2 and two distinct non-reserved colors. The resource
numbers alone do not prove queue capacity, ordering, liveness, or safe coexistence.
Those require compiler and runtime probes before use.

## SDK mechanisms to reuse first

The installed `<collectives_2d/pe>` already coordinates operation completion with
teardown and a blocked/activated lock task. Its root sends teardown control
wavelets after broadcast data; reduction has a network-specific teardown state
machine. This is a concrete SDK design reference, not a generic global barrier
that can be inferred from a local callback. Existing resident-solver work in this
repository also uses SDK collectives and records its queue reassociation limits
in `RESIDENT-SOLVER-RESOURCES.md`.

Before adding a new control protocol, evaluate whether a teardown-aware grouped
collective or an SDK collective schedule can provide the required ownership
transition with acceptable numerical order and cost. SDK collectives use two
queues in each bank per dimension; the current grouped collective uses five,
leaving only queue2 beyond default memcpy. They therefore cannot simply be
imported together with their default assignments. Safe phase reuse or a changed
communication schedule requires explicit validation. No active application is
changed by this design investigation.

## A fallback correctness baseline to investigate

A static alternating-color control cycle through the compute rectangle can use
one receive queue per PE: even/odd PE parity chooses the incoming color, and the
other color is used for transmission. This creates two code specializations,
without assuming one queue can simultaneously bind two colors. Keep this control
routing separate from the data colors that change axis.

A candidate protocol has three token rounds:

1. Collect readiness only after each PE has finished the old data operation.
2. Reconfigure data routes and acknowledge that every PE has installed them.
3. Release the next operation through the unchanged control plane.

A single token must fit the receiver's queue without needing the receiving PE to
leave the old operation. The root must consume the returning release token before
reusing the control epoch. All old data destinations/forwarding endpoints must be
accounted for; an acknowledged source send alone is insufficient.

This cycle is O(number of PEs) per round and is only a possible correctness and
stress-test baseline. Before considering it for production, compare existing SDK
synchronization facilities and a hierarchical protocol, quantify latency, test
injected per-PE skew, and prove absence of interaction with the old dataflow and
SDK command queues. Do not remove the protocol on an assumed timing margin.

The compiler-facing contract should include tensor ownership before/after the
transition, old and new routes, dedicated control resources, local completion,
global readiness, global installation, epoch progression, and the point at which
buffer/DSR leases may be reused. Final application qualification still requires
original-input numerical checks, actual CSL execution, warm calls, and scoped
performance evidence.

## Prepared bounded experiment

`experiments/axis_handover/` now contains an isolated control-ring baseline and
a preparation/execution driver. No application imports it. The latest prepared
g2/g4 bundles are `axis-handover-g2-20260907T184853277152Z` and
`axis-handover-g4-20260907T184853344856Z`, both under evidence. They are not yet
SDK execution evidence. Old prepared versions remain preserved.

The implementation separates begin/end around the caller's route installation.
A non-root forwards ready then waits for the root's configuration token before
returning from begin. Only end forwards configuration after local installation;
root completes that round before initiating release. This distinction prevents
early configuration when another PE has not yet finished old data work.

The probe uses64PEs, colors10/11 and input/output queue2 in addition to the
existing five-color data plane. Eight calls alternate Y/X/Y reductions and
2/8/2 extents with old-operation and pre-installation skew. Independent stdlib
binary16 expected words, final token sequence, actual delay values, queue masks
and max-PE intervals are preserved. Abstract topology/FIFO state review supports
the design only under its assumptions; actual CSL compilation and execution
remain required, followed by cost comparison before production use.

## Executed alternatives and selected next integration (2026-09-07)

The static ring baseline passed full eight-call probes at
`evidence/axis-handover-g2-20260907T184853277152Z` and
`evidence/axis-handover-g4-20260907T184853344856Z`. Its max-PE totals were
66223–67981 cycles including data operations and deliberate skew. It remains an
experimental correctness baseline, not a production application dependency.

SDK independent X/Y planes passed the same input/skew probes at
`evidence/sdk-independent-planes-20260907T191700814423Z` and
`evidence/sdk-independent-planes-20260907T191717108307Z`, with7360–7934 max-PE
cycles. They widen half input to f32, use SDK reduce/broadcast, then narrow once.
5120 of6144 words differ from the grouped-half baseline. The frozen comparison
reports explicitly retain this precision/schedule distinction; no pure barrier
or HLS application performance claim follows.

`toolchain/runtime/sdk_axis_reduce.csl` now provides the reusable single-flight
callback interface. Its full SDK probe192718 validates odd extents3/11/5,
nonzero tail canaries, immutable input, in-place and distinct destination storage.
X uses colors0/1, queues2/4 and tasks14/15; Y uses colors4/5, queues3/5 and
tasks16/17. Caller callback task10 is separate. The two f32 workspace arrays cost
8*capacity bytes per PE. No old data plane is reconfigured to another axis.
Completion is local SDK operation completion, not a global barrier.

The next application integration is a resident B5/N256/F512 FFN with explicit
local-f16/collective-f32 source policy. The SRAM estimate and numerical gates
must include all three matrices, observation storage and both SDK planes.
