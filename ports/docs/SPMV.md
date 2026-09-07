# Canonical CSC train SpMV

`mesh_spmv.v1` expresses `y=A*x` using the official SDK hypersparse algorithm:
north/south vector trains, per-partition sparse column products, then east/west
sparse row merge/reduction. This is a distributed sparse schedule, not a dense
mask representation. Current HLS cases describe 512×512 with4096 stored entries
on16PEs and4096×4096 with32768 stored entries on64PEs.

```cpp
#include "spatial.hpp"
void design() {
  auto values = spatial::input<4096,1>("values");
  auto row_indices = spatial::index_input<4096,1>("row_indices");
  auto column_offsets = spatial::index_input<513,1>("column_offsets");
  auto x = spatial::input<512,1>("x");
  #pragma csl dataflow rows=4 cols=4 storage=csc exchange=trains reduce=sparse_rows nnz_per_pe=512 cols_per_pe=128 rows_per_pe=128 fp=relaxed
  auto result = spatial::spmv_csc<512,512>(values,row_indices,column_offsets,x);
  spatial::output("result",result);
}
```

## Storage and semantics

The frontend uses real `uint32_t` index tensors. The checked IR records `u32`
for index ports and `f32` for numerical ports. Native serialization uses an
explicit `@u32` record and parses decimal integer tokens without converting
through float. Tests execute the C++ runner with16777217 and4294967295, and
reject fractional, negative and overflowing indices.

CSC is zero based, with monotone offsets, terminal offset equal to the number
of stored entries, and strictly increasing row indices in every column.
Unsorted rows and duplicates are rejected; no hidden sorting or duplicate
summation occurs. Values must already be finite f32 and satisfy the declared
input bound. Explicit zeros, including signed zero, remain stored entries.
The first frontend profile has a positive static number of entries; entirely
empty CSC is supported by the host storage helper but not yet a zero-extent
C++ tensor. These are different coverage claims.

Each PE receives sparse columns, a sorted compact row set, and u16 positions
into that row set. Global u32 indices are checked before local narrowing.
`nnz_per_pe`, `cols_per_pe`, and `rows_per_pe` are allocation capacities, not
assumptions that every PE has that occupancy. Every epoch is checked against
all capacities and the SDK-derived memory estimate plus an8192-byte control
reserve before compilation/execution. Compiler memory acceptance remains
separate. Positive backing storage permits empty partitions without inventing
nonzero entries. The original benchmark's optional randomization of matrix
values is not part of this operator.

`sparse-packing.json` preserves every epoch's structural lowering. The tests
compare all packed buffers to pinned SDK `preprocess.py`, and separately
reconstruct original entries and vector-padding boundaries. Input-vector pad
values are ones as in the original host; no packed sparse column may address
padding. Output padding must remain exactly zero.

## Source reuse and target integration

The source is SDK examples commit4866cf330333446cb5e529e10f36be4600d1df29,
`benchmarks/spmv-hypersparse/src`. Original files remain read-only. Adapted CSL
library templates retain license headers and are separately recorded in the
adapter patch and source manifest. High-level scheduling, storage packing,
host transport and numerical checking are separate Python modules.

Actual SDK compilation discovered input queue 1 is reserved by current memcpy.
The current adapter uses RX queues 4/2/6/7 and fixed north/south/west/east TX
queues 2/3/4/5. Send microthreads 0/3 are assigned explicitly; receive
microthreads follow their input queues. Phase and DSR ownership are checked
jointly by the planner. Compiler acceptance alone does not establish protocol
correctness.

The old benchmark-only allreduce clock synchronization is isolated as an
unfinished migration experiment. SpMV itself now uses SDK local timestamps.
The unsigned reverse-column scan also requires an occupied-extent guard before
metadata access; this is a source boundary repair, separate from SDK API changes.

SDK run `run-20260906T123640195928Z` passes both declared sizes, four calls
per size in a single runtime. Source-native performance comparison and a targeted large-case repeat are complete.

| Matrix / stored entries | PE mesh | Checked output values | Maximum absolute error | Estimated bytes per PE, including reserve |
| --- | --- | --- | --- | --- |
| 512×512 / 4096 | 4×4 | 2048 across four calls | 2.738e-7 | 16640 |
| 4096×4096 / 32768 | 8×8 | 16384 across four calls | 5.377e-7 | 34304 |

The successful adapter stages each outgoing row slice in u32 backing before
sending its exact logical number of u16 values. Two direction-specific buffers
cost `8*ceil(rows_per_pe/2)` bytes per PE and are not reused until the matching
send callback. Source offset plus count is checked in u32 arithmetic against
capacity before staging. Values remain in the original numerical buffers.
The SDK ELF/LMA audit confirms every application PE's staging alignment and
extent; the large mesh has20 shared ELF classes representing64 PEs.

This adaptation follows a reproducible odd-offset mixed-width transport
mismatch in the original path, not an assumed cause based on syntax. The
[isolated transport probe](../experiments/metadata_transport/README.md) records
24 successful alignment/packing cases and preserves failed controls. The exact
production reverse-scan function also passed a separate SDK probe for an empty
partition, a one-column partition, and a scan through index zero in two segments
(`reverse-scan-20260906T124212954571Z`). Earlier queue and auxiliary-clock
failures remain in WORKLOG and their immutable runs.

## Numerical and diagnostic contract

The independent numerical oracle accumulates the original CSC entries with
`math.fsum`, without reading compiler-packed tile rows. Every output must meet
fixed `rtol=3e-5, atol=3e-6`; zero-valued epochs must produce exact zero.
Changed values, vectors, structural indices, empty rows/columns/partitions and
uneven occupancies are exercised across four calls in one runtime.

Two numerical witnesses per PE retain the first/last occupied row's local
partial sum before east/west reduction reuses scratch storage. They are samples,
not a full internal numerical history. Completion diagnostics expose remaining
train/compute counts plus completed invocations. Saved runtime operation stages
identify a host call being attempted; a returned `launch` alone is not proof of
completed device work. A successful dependent D2H is the observation barrier.

The per-PE SDK timestamp interval excludes host transfers and is not a
synchronized global interval. No synchronized timing claim is made for SpMV. A source-native same-input
performance comparison is complete for these two cases. Arbitrary
sparse formats, duplicate canonicalization, graph composition with iterative
solvers, and full application-level sparse workloads remain subsequent work.

## Scoped comparison and floating-point order

Both source-native comparisons pass all four original-CSC accuracy checks. They
use the same adapted kernel, input, queue mapping, aligned staging and SDK local
timing, with HLS partial-sample arithmetic/capture removed. The adapter is not
compared to an unmodified legacy path that fails to execute on this target.

| Case | Native evidence | HLS/native maximum local interval over four calls |
| --- | --- | --- |
| 512² | native-spmv-20260906T124246960873Z | 1.03094–1.03315 |
| 4096² | native-spmv-20260906T124341222458Z | 1.02810–1.04074 |

These ratios measure bounded diagnostic overhead. They exclude host transfers,
are not synchronized global latency, and are not physical CS-3 performance.

Small-case outputs are bitwise equal. Large-case epoch0 differs on one row by
one ULP; epoch3 differs on42 rows (maximum absolute2.384186e-7, relative3.117003e-6,
32 ULP near a small result). Epochs1/2 are bitwise equal. Both independently meet
the unchanged original-entry tolerance; no tolerance was increased.

`experiments/explain_spmv_rounding.py` derives local north/south assignment
alternatives and the two directional accumulation orders from the source, then
enumerates their interleavings using original CSC entries. All43 differing rows
of both executions lie within the resulting f32 value sets. The model includes
separate and fused local multiply-add allowed by the declared relaxed policy;
it is not a trace proof of the exact executed association or a race-freedom proof.
The source performs east/west output additions in serialized local tasks.

A targeted same-input large HLS repeat, `run-20260906T125150595195Z`, passes all
four full-output/partial/progress checks and returns bitwise identical outputs
to `123640195928Z`. This supports stability for the tested setup; the public
contract remains numerical equivalence, not universal bitwise determinism.

## Synchronous initialization and task budget

`run-20260906T132458488734Z` passes512²/16PE and4096²/64PE, four SDK calls each,
with initialization invoked synchronously before activating the receive tasks.
The library now binds13 local tasks, releasing task10. Train arithmetic and
asynchronous callbacks remain in the library. Native regression132429291732
passes both profiles. The original14-task implementations remain in old runs.

`experiments/compare_spmv_initialization.py` independently re-audits disposable
copies of both frozen implementations before comparing identical inputs. It
preserves all old run files. The first comparison script assumed a literal host
output name `y` and stopped; the corrected script obtains the port from semantic
IR. This was an analysis-script error, not a numerical failure.

The two `evidence/mesh_spmv_*-synchronous-init-comparison.json` files record
per-call maximum PE-local cycle ratios and any changed f32 outputs. A change in
asynchronous task scheduling can alter relaxed accumulation order: bit equality
is reported, not required in place of the original-entry numerical checks.
These are simulator local intervals, not globally synchronized hardware time.

Measured after/before maximum-local ratios range0.99906–1.00438 for512 and
0.96834–1.02677 for4096 across four calls. No uniform speedup is claimed. The
largest output change is4.76837e-7; both frozen original-entry audits pass.
