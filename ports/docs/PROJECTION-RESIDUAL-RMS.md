# Projection, residual addition, and row normalization

This is the next source-backed composition under development. The first64×64 sampled HLS profile is qualified; larger/lean profiles are still being tested. It is not a full Prefill model. The preceding rectangular MLP precision work now has two complete blocked-accumulation qualifications.

The target expression is `z = activation * output_weight + residual; output = RMSNorm(z, gamma, epsilon)`. Its source is MeshInfra/WaferLLM commit `fd1c2daae37cd68706c03fc8009887ecee9900f8`, Apache-2.0, `Prefill/src/prefill.csl`: `h1_matmul`, `z_add`, and `rmsnorm_z`. Activations represent supplied attention outputs; generating those inputs is outside this isolated composition.

The source projection uses forward alignment and two-hop double-buffered communication. After its final compute/communication join, residual addition consumes a logical row/feature tile. Local row squares then feed the existing CSL row collective. Inverse scales belong to rows; gamma belongs to feature columns. None of these intermediate tensors should require a host transfer.

## Source experiments

`experiments/projection_residual_rms_source.py` prepares separate, immutable source experiments for:

- Original numerical functions with a valid zero stale-descriptor seed.
- An explicit correction of the descriptor loaded for the local square sum.
- That correction plus explicit row-vector inverse scaling.
- The extracted local CSL library with the original communication collective.

The seed is a harness choice: `rmsnorm_z` sets `seqLen_dsd_1` to square scratch but originally loads `seqLen_dsd_2`. Binding the latter to a sufficiently sized zero buffer isolates the stale-state dependency without an invalid one-element dummy extent. This is not an unmodified full Prefill execution. The second issue is independently isolated: original `local_sum[feature]` scales a contiguous row vector with one scalar, while row normalization requires that vector's individual row scales.

Three changed/zero input calls use independent weights and nonuniform residual rows/gamma. The analyzer checks source projection, residual, local/reduced sums, inverse and final output bits separately from original-input `math.fsum` projection and mathematical RMS normalization. Source behavior matching is distinct from mathematical acceptance. An intentionally defective source control may complete SDK execution and still fail the mathematics.

The library control `projection-residual-rms-source-20260907T062553956089Z` passed SDK2.10.1 compiler-only check `compile-check-20260907T063842493830Z`. The four source controls have now executed; generated-HLS execution is underway.

## Reusable local CSL interface

`toolchain/runtime/rms_local.csl` exposes three synchronous functions:

| Function | Caller-owned storage and completion contract |
| --- | --- |
| `square_sum(input, scratch, sums)` | Column-major local tile; scratch and sums must be disjoint from each other and the input. Writes half squares and half row sums. |
| Row collective in caller | Uses the established CSL communication library; completes before inverse calculation. |
| `inverse(sums)` | Replaces reduced sums with SDK half reciprocal square-root scales using global feature count and epsilon. |
| `normalize(input, gamma, output, inverse_rows)` | Applies feature gamma then row inverse with the source's half rounding sequence. Output may exactly alias input; gamma and inverse rows must remain disjoint from written storage. |

The module allocates no colors, queues, tasks, or microthreads. It leases explicit destination/source0/source1 DSR banks1 and2 only during synchronous calls. The caller must join earlier asynchronous users before entry and preserve the row collective's own lease. Compiler/SDK temporary register allocation is separate from these explicit leases and must be verified in the composed execution.

Static parameters require positive dimensions, local element count at most32767, global feature count at most2048, and positive half epsilon at most1. Input/gamma finiteness and finite half square-sum accumulation remain caller preconditions; the HLS composition must prove suitable range bounds. These shape checks alone do not establish numerical accuracy or dynamic stack safety.

The frontend's typed `spatial::add` now retains float32 or half tensor type. Native half addition rounds once from an exact-enough double sum, with tests for tie-to-even, cancellation and subnormal addition. Its existence does not imply that an arbitrary half graph has a qualified CSL lowering; graph and backend support remain explicit.

## Integrated development candidate

`projection_residual_rms_64x64_8x8/run-20260907T071024411920Z` now contains the executable C++ expression, all intermediate IR, resource/lifetime schedule, generated CSL and frozen implementation. Eight actual-native original-input checks pass (maximum relativeL2 about0.001132), including changed weights, residual-only, projection-only, bounded maximum operands, exact cancellation and a subsequent zero reset. Compiler-only SDK2.10.1 check `071236026140` accepts the generated CSL. Its eight-call simulator validation has completed and is registered in `qualification-20260907T074658560657Z.json`.

The shared `rms_bounds.py` computes the source half local/tree sum bound and enumerates every representable nonnegative reduced sum up to that bound through the validated SDK inverse model. This avoids assuming monotonicity of an approximate square root. The result is a conservative range bound, not a relative-accuracy proof. The64×64 candidate plans20928 bytes/PE including reserves; linked ELF and dynamic behavior still require executed validation.

The ordinary add policy is `#pragma csl dataflow rows=8 cols=8 partition=tiles compute=dsr fp=relaxed`. The verifier follows projection/add/normalization edges, accepts either add operand order, and rejects mixed regions, unsupported precision policies and unsafe range/memory configurations. The SDK transport is shared in `half_region_runtime.py`; it performs packing/transfer/decoding only. Target trajectories, protocol checks and independent application mathematics remain separate.

Debugger steps0..P−1 inspect projection prefixes, P inspects the residual sum, and P+1 inspects normalized output plus row inverse bits. Lean mode does not invent the unobserved matrix prefixes or residual tensor; its final inverse vector is explicitly observed.

## Executed source findings

All four64×64 source controls completed three warm calls. `projection-residual-rms64-original-seeded-review.json` confirms the original seeded descriptor behavior but fails the original-input mathematics. Correcting only the descriptor reduces dense relativeL2 to about0.101–0.103, still outside the fixed0.02 gate. `projection-residual-rms64-both-repairs-review.json` and `projection-residual-rms64-library-review.json` pass both target bits and original-input math. The explicit-repair and extracted-library variants agree bitwise on all six stage arrays; see `projection-residual-rms64-repair-and-library-control.json`. This establishes local library behavior in the actual composed source runtime, not general alias/concurrency guarantees.

The larger rectangular128×256 candidate `run-20260907T073732225890Z` passes eight actual-native original-input checks (maximum relativeL2 about0.002331). Its planned42144 bytes perPE include conservative reserves; linked memory and SDK execution remain pending. Corrected source controls use independent row/feature dimensions. Original feature-index diagnostic controls are intentionally restricted to square geometry because their wrong indexing otherwise may exceed row storage.

## First generated-HLS qualification

The64×64 sampled profile completes eight calls in one SDK runtime. All372736 sampled half intermediate values, final outputs, input immutability, row inverses, task progress and queue-drain checks pass. Maximum original-input device relativeL2 is0.00123613 under the unchanged0.02L2/0.03peak contract. `projection-residual-rms-audit-mutations-20260907T074522322314Z.json` rejects corrupted outputs at every epoch, each projection prefix, residual/local/reduced/inverse stages, input mutation, invalid raw data, timestamps and incomplete progress.

Static linked highwater is11296 bytes/PE, excluding dynamic stack. The matched repaired-source first three calls agree bitwise on final projection and normalization stages; HLS maximum local intervals8902–8910 versus8460 source include all-prefix/operand observation and immutable-copy differences. This approximately5.2–5.3% sampled overhead is not a hardware performance measurement. The lean counter candidate074355434925 will measure the distinct observation policy and is not yet qualified.

Full121-profile native regression `run-20260907T073733216957Z.json` and202unit tests pass after frontend/middleware integration; the newly registered composition will also enter the next full catalog regression. The debugger's `--check-completed` output explicitly labels partial diagnostics as not full qualification. A preserved actual run diagnostic is `projection-residual-rms64-completed-debug-074310.json`.

## Range analysis for further composition

`rms-correlated-range-metadata-review.json` records a tighter compiler range proof. Every nonnegative half row/tree sum is at least each participating rounded square. For each possible half input magnitude, inverse suffix maxima enumerate every representable permitted sum above that lower bound, followed by the source's gamma-first half rounding. This covers squaring underflow without assuming approximate SDK sqrt monotonicity. The64/128 feature profiles' output bounds tighten from2234/6704 to12.015625/24.03125. These are conservative ranges, not numerical-accuracy guarantees.

All three prepared profiles regenerate byte-identical CSL and notices with this metadata change; historical schedules remain frozen. Existing evidence/debugging must use each bundle's preserved implementation when regenerating historical metadata. This proof prepares bounds for longer resident compositions; it does not itself add full model support.
