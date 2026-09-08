# Supplied attention-output projection and normalized FFN tail

This kernel composes the pinned WaferLLM Prefill `h1_matmul → z_add → rmsnorm_z → z1_matmul → z2_matmul → z3_comp → h2_matmul → add_result` boundary. The source is MeshInfra/WaferLLM commit `fd1c2daae37cd68706c03fc8009887ecee9900f8`, Apache-2.0. Upstream files remain read-only. This starts with supplied attention output; it is not complete Prefill, Decode, attention masking, multihead orchestration or a KV cache.

## Frontend semantics

Seven ordinary typed inputs describe attention `[M,N]`, output weight `[N,N]`, original residual `[M,N]`, feature gamma `[1,N]`, up/gate weights `[N,F]` and down weight `[F,N]`:

```
projection = attention * output_weight
Z = projection + residual
X = RMSNorm(Z, gamma)
U = X * up_weight
G = X * gate_weight
delta = (U * SiLU(G)) * down_weight
Y = Z + delta
```

The final residual is **Z**, not the original residual input, and no normalization follows the final addition. The graph verifier follows typed edges, accepts commuted additions and renamed identifiers, checks all seventeen nodes are consumed, and rejects substituting the wrong residual. This is a bounded structural lowering, not general C++ synthesis or arbitrary graph fusion.

Pragmas expose forward two-hop exchange, double-buffered transfers, local DSR computation and an 8×8 PE region. The output projection retains the source half recurrence. Up/gate/down explicitly request block-f32 accumulation with a block equal to the physical K tile. The native typed operation and spatial accumulation order can differ under declared relaxed arithmetic; both must independently satisfy fixed original-input accuracy limits. No tolerance is inferred from a successful range check.

## Shared CSL implementation

`mesh_prefill_tail.py` composes existing projection/residual/RMS and FFN plans. `prefill_tail_csl.py` adds a joined prelude to the shared projection engine through typed region hook slots. It does not duplicate the matrix transfer engine or add application colors, queues, local tasks or microthreads. The same projection callbacks finish both communication axes and local compute before the residual/RMS phase begins. The original row collective, `rms_local.csl`, `gated_local.csl`, `block_accumulate.csl`, DSD views and explicit DSR leases remain the numerical/communication infrastructure.

The public attention, output weight and residual are immutable. Existing rotating left/right buffers serve the output projection, then serve the MLP after completion. A distinct private `post_projection_z` allocation survives through the final addition. Normalized work storage may become the down result only after both up/gate consumers complete. `prefill_tail_lifetimes.py` checks eleven phase boundaries and the declared physical allocations. This verifies scheduled lifetimes and explicit resource leases; compiler temporaries, SDK-managed resources and dynamic stack still require separate validation.

The first 64×64→256→64 profile has 7456 bytes of declared numerical storage per PE. Observer arrays, generated code and SDK allocations are additional; linked ELF measurement is required before registration. Host packing obeys the source block ownership. There is no intermediate host tensor transfer between the four projections, normalization, gating or residual operations.

## Reproducible validation

`experiments/build_prefill_tail.py` writes a source-described application folder and a fresh frozen bundle. Every bundle retains Clang AST, frontend/checked/optimized IR, semantic graph, schedule, native executable/stdout, generated CSL, compiler implementation, SDK settings and stage evidence. `HLS_CLANGXX` selects the same compiler for the AST and native executable; the SDK host uses `/usr/bin/clang++-17` because its default Clang14 cannot compile the x86 `_Float16` native runtime.

Before SDK, two separate diagnostic C++ builds expose the actual projection and down result using the preserved Clang declaration ranges. The original public outputs must remain exactly unchanged. `application_gate.py` hashes both observation bundles. Independent `math.fsum/sqrt/exp` checks evaluate the complete algorithm from all seven original inputs and separately enforce 2% relative L2 and 3% reference-peak error on projection, MLP delta and final output. A predicted-target gate follows. A final residual pass cannot hide a failed internal delta or projection.

On SDK, `mesh_prefill_tail_sdk.py` verifies every public input, projection/Z snapshots, the private prelude counters, the shared FFN protocol, normalized/down/final half values and all three f32 accumulator witnesses. Sampled mode additionally checks every projection prefix and first shifted operand. Counter mode retains final snapshots and protocol counters; it does not claim unobserved prefix values. Deliberate mutations must be rejected before registration.

The matched source control uses the same inputs, fabric, SDK options and explicit precision policy. It includes the already documented source RMS/gate ownership repairs. Observation and copy differences remain in local simulator intervals and must be stated. WSE3 simulator cycles are not CS-3 hardware throughput.

## Debugging

The completed-call debugger validates a single result snapshot using the bundle's frozen compiler/auditor. In the 8×8 profile, steps 0–7 expose output-projection prefixes in sampled mode; step8 is the final projection, step9 retained Z, step10 normalized X, steps11–35 the shared MLP view, step36 delta and step37 final output. Counter mode reports missing prefixes as unobserved. Every prelude view includes its actual completion counters. Partial completed-call validation is not full-run qualification.

## Development evidence

- `prefill-tail-source-20260907T100400857864Z`: three executed source calls, independently checked in `evidence/prefill-tail64-source-executed-review.json`.
- Counter HLS `run-20260907T102005944852Z`: eight native/target gates and eight SDK calls pass. Registered by `qualification-20260907T103428102484Z.json`; 54 corruptions rejected. Static ELF21712B/PE; unallocated27440B. Three matched source ratios1.01000–1.01115 include immutable-input and observation differences. Maximum original-input relative L2 projection .00120310, delta .00537423, final .00114027.
- Sampled HLS `run-20260907T102500202051Z`: eight native/target gates and eight SDK calls pass; 82 corruptions rejected. It observes3,670,016 internal half values and294,912 f32 accumulators. Matched-source local ratios1.03439–1.03817 include prefix observers. Static ELF36656B/PE; unallocated12496B.
- The full226-test suite and the subsequent seven affected tail tests pass (227 distinct tests after adding the early-debugger regression), including exact legacy FFN CSL generation, structural mutations, live-Z alias rejection, outstanding resource release rejection, debugger observation scope and mandatory internal numerical checks.

Larger128×64 counter HLS `run-20260907T102725432573Z` passes eight native/target gates and eight SDK calls;54 corruptions rejected. It observes589,824 f32 accumulators. Matched-source local ratios1.00622–1.00710; registration qualification-20260907T110057556215Z.json. Static ELF30272B/PE; unallocated18880B. Its matched source control `prefill-tail-source-20260907T102741323756Z` completes three calls and passes the original-input mathematical review.

All three profiles are registered after complete SDK calls, corruption rejection, matched source comparison and linked memory. The same64sampled/counter outputs, retained snapshots and three f32 accumulators are bit-identical across all eight calls; source and templates differ only by observation mode. Separate fresh SDK-host Clang17 native builds of all three frozen bundles pass eight actual projection/delta/final checks and reproduce the executed CSL bytes. Evidence is retained in `evidence/prefill-tail-native-clang17-20260907T1100Z/`. No full-model or hardware claim follows from this boundary.

## Following connection: supplied Q/K/V attention

The pinned source dispatches score contraction, scaled softmax and value contraction immediately before this tail. The existing eight-node resident attention boundary provides a separately executed control. Connecting it must preserve that algorithm's key-transpose descriptor, softmax row max/sum completion, device-side V alignment and strided right-operand DSD. The tail's first ordinary projection must reset the right descriptor back to contiguous access. Existing attention already tests its score/value entry reset over repeated calls; extending the chain must test the additional value-to-output-projection transition.

The same colors/queues/tasks cannot be reserved independently by two simultaneously active engines. A composed schedule must join attention's score/softmax/value phases before the tail acquires those leases, and model dead K/score scratch separately from live attention output and final Z. Preserve original residual and all weights. Derive the attention-output range from the probability mass and value bounds; do not silently substitute a supplied-input bound for a computed value. Independently check attention output, output projection, MLP delta and final output so residuals cannot conceal a failed earlier stage.

This next connection is a pending implementation. It still would not include the source's preceding input RMS, Q/K/V projection, two RoPE operations, head replication, masks or cache behavior. The original dispatch retains these as distinct stages; coverage must name exactly which are executed.

## Developer entry points

| Layer | Entry point and retained evidence |
|---|---|
| Application authoring | `experiments/build_prefill_tail.py`; per-profile `hls.cpp` and `PORT.json` |
| C++ frontend | Clang AST and `00_frontend_command.json`, then `01_frontend_ir.json` |
| Typed composition | `mesh_prefill_tail.verify`, prefix/core contracts, checked and optimized IR |
| Spatial scheduling | `mesh_prefill_tail.plan`, `schedule.json`, numerical bounds and eleven-phase storage/resource plan |
| CSL generation | `backend.generate` → composed FFN/projection engine hooks; `layout.csl`, `pe.csl` and reusable CSL modules |
| Native branch evidence | `native-projection-observation/` and `native-delta-observation/`, plus `application-gate.json` |
| Target preflight | `target-application-gate.json`; predicted arithmetic, explicitly not device observations |
| Actual SDK | Frozen `sdk-execution-driver.py` and `implementation/sdk.py`; compiler command, SDK log, stage records and raw `results.json` |
| Device audit | Frozen `mesh_prefill_tail_sdk.audit`; decoded outputs plus exact raw staged observations |
| Qualification | Mutation report, matched source comparison, linked ELF report, `register_prefill_tail.py` and qualification index |

Build with `python3 experiments/build_prefill_tail.py --instrumentation counters --geometry 64 64 256 8`; use a fresh bundle for every execution. `--instrumentation sampled` retains all internal prefixes. Inspect a saved bundle with `python3 toolchain/debug.py <bundle> --node p0_0 --epoch 0 --step 36 --check-completed`. Before the first saved call, the debugger reports zero completed calls and unavailable evidence rather than claiming a successful audit. The SDK execution driver refuses a missing/failed native or predicted-target branch gate.
