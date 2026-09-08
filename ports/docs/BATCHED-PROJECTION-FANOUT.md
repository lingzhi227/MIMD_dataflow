# Batched normalized projections on the Decode layout

This is a source-backed **linear-algebra subgraph**, not a complete inference
engine. The frontend expresses an RMS-normalized tensor feeding two or three
independent matrix products. The implementation follows the resident batch and
fused projection reduction in the pinned WaferLLM Decode source. It is separate
from the earlier tiled Prefill fanout, which has a different communication and
buffer-ownership algorithm.

## Layer contracts

| Layer | Responsibility |
| --- | --- |
| C++ frontend | Typed inputs, ordinary RMS/matmul expressions, public branch outputs, explicit numerical/dataflow policies |
| Structural IR | Follow SSA edges to find the shared producer and distinct weights/outputs; independent of application names |
| Physical planner | Feature and weight ownership, branch offsets, finite ranges, padded collective extents, PE memory and register leases |
| CSL local library | Memory-DSR RMS accumulation, SDK half square root, gamma-first normalization, row-major DSR/map matmul |
| CSL communication library | One five-color grouped Y instance; synchronous calls with changing DSD lengths and unchanged routes |
| SDK transport | Original inputs enter through memcpy; compiled CSL executes; actual outputs and diagnostics are read back |
| Audit/debug tools | Exact target-order words, independent original-input mathematics, replica/progress/queue checks, scoped partial inspection |

No numerical oracle or upstream authoring script participates in CSL generation.
The native C++ path and independent mathematical checks remain separate from the
target-order model. The latter models a deliberately relaxed binary16 schedule;
it is not the mathematical correctness oracle.

## Ownership and fusion

The input has shape `[B,N]`, gamma `[1,N]`, and each weight `[N,F]`. A PE at `(x,y)`
owns every batch row for a contiguous input-feature shard on Y. Its weight tile
owns input features on Y and output features on X. After one RMS reduction and
normalization, all branches reuse the resident normalized input. Local output
storage is `[branch][batch][local output feature]`. One reduction of that entire
vector replaces separate branch reductions. The final output-feature shards are
on X and are replicated across Y rows.

The first bounded lowering supports two/three branches with the same `F`, square
4/8 meshes, and two-level midpoint groups of size2/4. `B` is independent of mesh
size. An odd RMS batch vector gets one zero padding lane; logical batch count and
projection shape are unchanged. Output shards must have even length. Rectangular
weights are allowed. Different branch output widths, cache mutation, GQA/head
placement, and in-call route-axis switching are not currently admitted.

The projection directive is:

```cpp
#pragma csl dataflow rows=8 cols=8 broadcast=resident_rows reduce=grouped_two_tree groups=2 result=feature_columns replicas=rows fusion=collective compute=dsr fp=relaxed
```

It accompanies the batch-major feature-Y RMS policy shown in the application
sources. Unknown, incomplete, inconsistent, or unsupported policies fail closed.

## Floating-point and protocol evidence

The requested epsilon is a frontend scalar. For `math=sdk_half`, CSL receives its
binary16 encoding; e.g. `1e-6` encodes as word17, approximately1.01327896e-6. The
independent oracle retains the requested mathematical epsilon. Tiny input cases
therefore matter: matching the target's own rounding model alone is insufficient.
The fixed per-branch acceptance limits are relativeL2≤0.02 and
peak-scaled-error≤0.03; this is explicit reduced-precision policy, not a guarantee
of float32 accuracy. Every branch is checked separately, including zero/tiny,
signed, structured, impulse, and changed-weight calls.

Actual SDK observations revealed that the Decode even midpoint root receives HEAD
on rd0 and TAIL on rd1, and consumes HEAD first. This differs from the earlier
standalone grouped-GEMV layout's association. `decode_grouped_reference.py`
models the actual Decode routes. It is shared by batch-major RMS and fanout.
The original135 RMS observations still pass the corrected model and unchanged
schedule. Older frozen evidence retains its own implementation.

Two directed SDK probes each execute eight warm calls with lengths2→8→2 on the
same communication instance. Groups2/group-size4 and groups4/group-size2 cover
both tree levels. Positive and signed fixtures distinguish opposite half-add
association; a separate stdlib-half review finds640 distinguishing lanes per
probe. See the preserved `decode-grouped-order-g2/g4-20260907T171040...` evidence.
These primitive checks are not a substitute for full application qualification.

## Resource and completion boundaries

The instance uses colors5–9 and input/output queues3–7. Local routines explicitly
reserve dest/src0/src1 bank1 and, for scalar RMS accumulation, dest/src0 bank2.
The collective uses **src1 bank2**, a different register. Phase leases record bank
and index separately. Calls are synchronous and branches execute sequentially;
no asynchronous overlap is claimed. SDK command/memcpy tasks and compiler
register temporaries are outside this explicit user-register inventory.

Changing a DSD length after a local collective returns is not evidence of a
global barrier. Every PE follows the same length sequence, and routes stay Y.
A blocking SDK command completes across the region before the next host call.
No in-call X/Y reconfiguration is generated.

## Debugging and qualification

Each fresh run preserves AST, typed IR, schedule, native executable/stdout,
generated CSL, implementation hashes, SDK commands/logs, and raw device words.
Debug steps0/1/2 are RMS local sum/reduced sum/normalized tensor. Steps3/4 are
packed local/reduced projection branches and show branch offsets, batch count,
feature ownership, and replica coordinates.

```sh
python3 toolchain/debug.py RUN --node p7_7 --epoch 7 --step 4 --check-completed
```

A frozen auditor which supports partial inspection can validate saved complete
calls with `require_complete=False`. It separately reports `full_run_passed` and
`expected_epochs`; partial success never registers an application. Old frozen
full-run auditors keep their original full-batch requirement.

Source-control qualification keeps original Decode matvec functions and fused
collectives, with explicit RMS recurrence/input repairs and padding only for the
RMS collective. The UP/GATE control directly binds the normalized-input subgraph;
it does not claim the preceding attention/residual computation. Comparison must
bind both result files and SDK image hashes, all branch/local words, independent
mathematics, linked ELF memory, mutation rejection, and actual SDK-host native
rebuild before catalog registration.

Simulator performance comparisons use max-PE cycle intervals including observer
stores and excluding host readback. Complete diagnostic readback is expensive in
the simulator and is not a production inference host API or a hardware throughput
measurement. Readback is retained for qualification.

## SDK observation experiments

In SDK2.10.1 live probes, `read_symbol` explicitly rejects a running simulation.
A live debug-util path returned stale zero values, and the live ELF-dump path did
not match the current input read through memcpy when interpreted with the SDK
ELF reader. Successful API return alone is not valid evidence of a current
snapshot. Those experiments are preserved, and **none is used to replace the
qualified observation channel**. They establish a negative result for the tested
configuration, not a general claim about every SDK debug workflow.

## Completed QKV configuration

B3/N512/F512, three branches on8x8PEs passed eight standard SDK calls and eight
source-control calls in one runtime per implementation. All10 raw port groups
match for every call; all48 injected faults are rejected. Original-input
independent maximum relativeL2 is0.00632074 and peak-scaled error0.00722428.
Max-PE cycles are13276–13279, only1.005604–1.005606 times the source control.
The projection portion represents4,718,592 multiply/add operations per call;
its aggregate rate is about355operations per max-PE cycle over the64PE rectangle.
That excludes RMS operation counts and is not hardware throughput/utilization.
Full diagnostic host calls took522–557seconds in this simulator configuration.
Linked static high-water is37168bytes/PE; no dynamic-stack measurement is implied.

See `evidence/batched-qkv-full8-source-comparison.json` and
`evidence/qualification-20260907T182741945418Z.json`. The catalog now has136bounded
qualified configurations; this does not mean136distinct complete applications.
UP/GATE B5/N256/F512 subsequently qualified as configuration137; details below.

## Completed UP/GATE configuration

B5/N256/F512, two branches on8x8PEs passed8standard and8source SDK calls.
All9 raw port groups match on every call,44corruptions are rejected, and the
frozen incremental checker records8completed calls with full_run_passed=true.
Independent maximum relativeL2 is0.00629146 and peak-scaled error0.00755919.
Max-PE intervals9245–9249cycles are1.007736–1.007740 times the source control.
Static high-water20672B/PE compares with20768B/PE for the source. Full diagnostic
host calls take210.8–213.9seconds in this simulator configuration. Both-host
actual C++ outputs pass and generated CSL bytes match.

See `evidence/batched-upgate-full8-source-comparison.json` and
`evidence/qualification-20260907T185614692044Z.json`. The catalog now contains137
bounded qualified configurations. The two new configurations share one structural
fanout lowering and the same local/communication libraries; they are not full
QKV attention or FFN/Decode application qualification.

Current authoring additionally uses the existing region_lifetimes verifier for
explicit numerical allocation lifetimes and local DSR completion tokens. Eight
focused and280full tests pass. Native QKV183602 emits the same CSL as the executed
QKV bundle. This metadata extension does not retrofit historical frozen runs.
