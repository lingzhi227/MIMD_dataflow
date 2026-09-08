# Next linear-algebra boundary: batched Decode primitives

Status: batched RMS3×512/8×8/g2, normalized QKV3×512×512 and UP/GATE5×256×512 now qualify as configurations135–137, each after eight standard SDK calls and matched source/native/mutation/memory checks. See `BATCHED-PROJECTION-FANOUT.md` for full measurements.
The qualified 31-node Prefill-shaped mixed chain remains separate.

## What the pinned source actually implements

WaferLLM `fd1c2daae37cd68706c03fc8009887ecee9900f8`,
`Decode/src/decode.csl`, implements batch-major local row-vector products and
reconfigures one five-color collective network between PE axes. Q/K/V local
products are contiguous in `QKV_tile`; one grouped reduction handles all three.
Up/gate similarly share a fused reduction. `score_matvec_mult` reads externally
supplied `XKCache_tile`; `output_matvec_mult` reads `XVCache_tile`. The inspected
kernel does not append newly computed K/V to those arrays. The two
`pes_p_head`/`pes_p_kv_head` parameters have no uses after their declarations in
this kernel. Do not infer implemented cache updates or GQA from parameter names.

The next faithful port is batched normalization and grouped projection/fanout,
then the read-only supplied-cache score/softmax/value path. It needs explicit
feature sharding, batch-major local layout, result replication and exclusive
route-reconfiguration phases. Actual head/cache update contracts require more
source evidence and distinct state semantics; they are not covered here.

## SDK discrepancy investigation

All reference source trees remain unchanged. Each probe copies the source,
records file hashes and a diff, and runs eight changed-input calls on an 8×8
mesh with batch size 2 and 64 global features. Gamma is nonuniform in later
calls. Device inputs remain immutable and outputs are checked on all PE replicas.

1. `evidence/decode-rms-source-20260907T153205576180Z`: original RMS and a
   version changing only the normalize input DSD from squared scratch to X.
   Both fail original-input mathematical RMS accuracy; failure is retained in
   `evidence/decode-rms-first-repair-failure.json`.
2. `evidence/decode-rms-source-20260907T153456870135Z`: additional device
   observations before/after grouped reduction. The first local batch sum is
   exactly the final feature square. The second batch sum is a full sum.
   The observed wrong value exists before communication, so route changes do
   not repair it. This is execution evidence, not a general claim about every
   scalar-form CSL reduction or an established compiler root cause.
3. `evidence/decode-rms-source-20260907T153656391963Z`: direct pointer-as-src0
   repair rejected by the compiler. The diagnostic enumerates legal `@faddh`
   operand signatures. This failed compilation is retained.
4. `evidence/decode-rms-source-20260907T153833031675Z`: stationary memory
   accumulator in dest/src0 DSR 2, one-element advancing src1 DSR 1, explicit
   zero and sequential feature additions. Together with the X input repair,
   this passes all eight original-input checks, including nonuniform gamma,
   zeros, negative and sparse data. Maximum relative L2 is 0.000573571 under
   the unchanged 0.01 gate. Original code fails seven nonzero cases. The same
   source grouped collective remains in both versions; Y/X/Y route resets
   are not replaced by Python. `review.json` binds actual results and frozen
   analysis driver. This is repaired source evidence, not HLS qualification.

Prepared-only `153632233819` was superseded before execution; it is not a pass.

The same executed probe samples original `fast_exp`: at -1 it returns
0.984375, while mathematical exp(-1) is approximately 0.367879. At -12 it
returns 0.824707. This source approximation cannot silently serve as a softmax
accuracy reference. The next HLS softmax must use an explicitly validated SDK
math policy, with accuracy and cycle costs reported separately.

## Compiler/runtime work required next

- Extend the existing typed RMS/collective policy with feature-axis sharding,
  batch-major DSD access and replicated outputs; batch size must not be forced
  to divide the mesh width merely to reuse the Prefill tile profile.
- Reuse the checked local RMS arithmetic and grouped routing infrastructure.
  Bind explicit memory accumulators and descriptor leases; distinguish local
  reduction completion from inter-PE reduction and host completion.
- Model the axis switch as exclusive reconfiguration after the prior collective
  joins. Audit queue ownership, replica equality and eight warm resets.
- Fuse Q/K/V and up/gate transfers only after their local producers complete,
  with a typed concatenation/partition contract rather than application-name
  dispatch. Preserve individual branch outputs for independent accuracy checks.
- Before admission, produce a new public HLS source and every standard IR/CSL
  stage, run actual native and SDK gates, compare corrected source behavior,
  inject faults, inspect linked memory and measure scoped device cycles.

Do not expand stencil or physics ahead of this linear-algebra queue.


## Shared CSL module verified

`toolchain/runtime/batched_rms_local.csl` now contains the batch-major local
stages (`square_sum` and `normalize`) with explicit DSR ownership and no allocated communication resources.
Probe `evidence/decode-rms-source-20260907T154512394628Z` actually imports this
module. All eight SDK calls match repaired source output and reduced sums
bitwise (8,192 output half words and 1,024 sum words), while the original-input
RMS gates still pass. `evidence/decode-batched-rms-library-eight-review.json`
binds the current module bytes to the executed copy. This is reusable runtime
foundation; the batched/grouped HLS frontend/profile remains the next task.


## Public batched HLS boundary qualified

`projects/waferllm/batched_rms_3x512_8x8_g2` now supplies the complete public
C++→Clang→typed IR→planner→CSL→SDK path. Its odd batch count exercises an explicitly
zero padded communication slot without requiring B%P=0. Qualification
`evidence/qualification-20260907T161741173353Z.json`: eight actual SDK calls,
source bit equality,28corruption rejections,ELF8832B/PE and1.36% max-PE simulator
cycle overhead. Prior134CSL configurations remain byte-identical under current
codegen. No axis switch inside a running collective or full Decode is claimed.
Next is a generic two/three-branch normalized projection with one packed grouped
collective, matching source QKV and up/gate packing rather than kernel names.

### Resident normalized projection fusion (in progress, 2026-09-07)

The next lowering canonicalizes a shared RMS producer and two or three independent
matmul consumers using SSA edges. `normalized_fanout_ir.py` owns this structural
check; the previous tiled Prefill lowering and the new batch-major Decode lowering
retain separate physical policies. No application or host-port name selects the
backend. The old normalized fanout regression tests still pass after extraction.

The new projection policy is:

```cpp
#pragma csl dataflow rows=8 cols=8 broadcast=resident_rows reduce=grouped_two_tree groups=2 result=feature_columns replicas=rows fusion=collective compute=dsr fp=relaxed
```

This policy requires the feature-Y/batch-major RMS producer. All batch rows are
resident on each PE. A weight shard owns input features on Y and output features
on X. Branch-local results are packed as `[branch][batch][local output feature]`
and reduced together along Y. Output rows are replicas; they are not additional
logical batches. Rectangular weights are supported when all branches share the
same output feature count. The planner rejects non-even output shards, mismatched
branch policies, unsupported aliasing, unsafe numerical range, and PE/DSD budget
overflow. The bounded first implementation uses 4/8 square meshes and no in-call
axis reconfiguration.

`batched_matmul_local.csl` uses the source Decode `vecmat_computation` DSR/map
pattern. `grouped_collective_csl.py` derives a dynamic-extent module from the
qualified static grouped collective without changing that static file. One
instance owns colors5–9 and queues3–7. Eleven DSD lengths (ten fabric, one memory)
change from padded RMS batch extent to fused projection extent. Every PE executes
the same ordered blocking sequence. Local return is not a global barrier; no
route switching is justified by this experiment.

Candidate application subgraphs:

- QKV: B3, N512, F512, three weights, 8x8PEs, two groups; RMS extent4 then fusion576.
- UP/GATE: B5, N256, F512, two weights, 8x8PEs, two groups; RMS extent6 then fusion640.

The frontend remains ordinary typed RMS/matmul expressions and public branch
outputs. Source controls retain the original Decode matvec functions and fused
QKV/ZZ reductions. RMS sum/input repairs remain explicit. Only RMS collective
DSDs are padded in the source control, so odd logical batches do not inflate the
projection extents. The UP/GATE control binds its normalized input directly to the
RMS subgraph; it does not claim to reproduce the preceding attention/residual
chain. No cache mutation, full Decode, or hardware throughput claim is made.

Initial compiler experiments rejected `packed` (reserved word), `@max` (unknown
builtin), and arithmetic on a pointer to an entire array. Generated code now uses
`fused_len`, a compile-time conditional, and explicit `@ptrcast([*]f16, &a[offset])`
following existing CSL code. These failed frozen builds are retained. The first
corrected QKV build compiled and launched in SDK2.10.1; full eight-call audits and
source performance qualification are still pending at this entry. Linked static
high-water is37168bytes/PE (11984bytes below48KiB), not a dynamic stack bound.

Debug steps0–2 inspect RMS local/reduced/normalized values; steps3–4 expose packed
local/reduced branches, with branch offsets and PE feature ownership. Frozen
completed-run inspection requires the full saved batch for this auditor.
