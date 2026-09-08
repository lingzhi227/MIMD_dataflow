# Half-precision two-hop matrix contraction

This lowering preserves the WaferLLM MeshGEMM spatial algorithm: bidirectional device alignment of X, host permutation of W block rows, a two-hop mesh cycle, two operand buffers, and asynchronous communication overlapping local DSR half FMA. It enhances CSL: the backend uses the original task/queue/microthread protocol and CSL operations rather than replacing them with host arithmetic.

## Frontend and numerical meaning

```cpp
auto a = spatial::input<128,128,spatial::f16>("a");
auto b = spatial::input<128,128,spatial::f16>("b");
#pragma csl dataflow rows=4 cols=4 exchange=two_hop initial_align=bidirectional reduce=local overlap=double_buffer fp=relaxed compute=dsr
auto result = spatial::matmul(a,b);
spatial::output("result",result);
```

The native C++ definition uses fused binary16 accumulation in increasing K order. The spatial implementation uses the same fused arithmetic in a cyclic block order. `fp=relaxed` explicitly permits that reordering; bitwise native/spatial identity is not promised for arbitrary dense inputs. The auditor checks every scheduled prefix bit, then an independent original-product error bound. Signed identity, zero reset, minimum subnormal and a fused-versus-split rounding fixture provide exact algorithm-level witnesses. Binary16 accumulation is not f32 accuracy; users needing a different accumulation type require a separately implemented and qualified lowering.

The target arithmetic probe `evidence/half-arithmetic-20260906T170538008876Z` executes 65 half entries through official SDK16-bit transport, direct half FMA, separate multiply/add and explicit DSR1 FMA, including changed warm inputs and subnormal/tie/cancellation cases. All expected bits match. It qualifies arithmetic/transport semantics, not a matrix kernel by itself.

## Layers and resource ownership

- `spatial.hpp` and the Clang frontend retain the tensor's half type. Unqualified half operations fail closed.
- `mesh_twohop.py` checks the typed contraction and constructs a dimension-based schedule. It does not dispatch by paper or application name.
- Four CSL runtime modules preserve the source task/route structure. DSR1 executes local FMA; communication uses DSR2–7 and microthreads1–4. Queues2–5 are reserved in both input/output banks; SDK queues0/1 are not reused.
- Host X/W pointers remain invariant and distinct from rotating working pointers. One runtime instance performs changed-input warm launches. Packed half tile extents must be even.
- The host uses SDK `input_array_to_u32` and MEMCPY16BIT; packing only permutes tiles. No host matrix multiplication contributes to the device output.
- The bounded profile supports square4/8 meshes, divisible matrix dimensions and signed DSD offsets. Static memory estimates include buffers, full prefix diagnostics and a code/task reserve; linked ELF inspection is recorded separately and is not stack high-water evidence.

Source reference: pinned WaferLLM `fd1c2daae37cd68706c03fc8009887ecee9900f8`, `projects/waferllm/upstream/MeshGEMM/src`. Original host permutation and hashes are preserved under `references/waferllm-host`. Tests independently compare packing to the original helper and cycle evolution to its `assignId` recurrence.

## Current execution evidence

64x64/4x4: `run-20260906T172641742457Z`, SDK2.10.1 six warm calls passed, 99,840 internal half observations (98,304 prefix values plus operand witnesses). Every scheduled FMA prefix is bit-exact; callback counts, epoch count and queue drain witnesses pass. Random output differs from native increasing-K by at most0.005859375, explicitly reported; five exact fixtures match. Linked static allocation is11,664 bytes/PE out of49,152; remaining37,488 is not measured stack headroom.

Thirteen adversarial mutations are rejected by the frozen auditor, including a one-bit prefix error, coherent final/output/history corruption, stale zero result, wrong operand block, missing callbacks/alignment, bad queue/timestamp and host lifecycle changes. Evidence: `twohop-audit-mutations-20260906T173703296535Z.json`.

`toolchain/debug.py CASE --node p3_2 --epoch 5 --step 3` reports original half bits and decoded values, expected K block, completed round counts and local timestamps. It reads stored device evidence without importing the SDK.

128x128/4x4: `run-20260906T173506392821Z` passed six SDK warm calls and 394,752 internal half observations. Linked static allocation is 25,488/49,152 bytes per PE. Random scheduled/native difference is 0.015625; all five exact fixtures match. Both 64 and 128 are now in the catalog. 256x256/8x8 `run-20260906T173947330774Z` passes six SDK calls and 3,158,016 internal half observations. Random scheduled/native difference is 0.03125; the original-product maximum absolute error is 0.024354632943868637. Linked static footprint is 33,760/49,152 bytes per PE. Source-native control `twohop-native-20260906T182417273560Z` passes two calls with bit-identical output and HLS/native maximum-local ratio 1.0325297433. The 256 profile is now in the catalog (76 total bounded profiles).

Original-source controls `evidence/twohop-native-20260906T174134550434Z` (64) and `twohop-native-20260906T174725559067Z` (128) retain original DSR compute and communication, with a documented logical-coordinate warm-entry and timing ABI adapter. Both changed-input calls produce bit-identical HLS/native outputs. Sampled HLS/native maximum-local interval ratios are 1.0885997522 and 1.0324207193 respectively. These exclude host I/O and are not hardware or synchronized global latency measurements.

## Optional diagnostic volume

`--instrumentation counters` omits full prefix stores and their host readback. Operand-corner witnesses, local timestamps, callback/epoch counts and queue status remain. The auditor still independently reconstructs the scheduled recurrence and checks every final half bit, but reports `prefix_observed=false` and `scheduled_prefix_bits_exact=null`; absent history is never counted as tested.

The matched 64 counter run `run-20260906T174509420723Z` passes six SDK calls with identical outputs and protocol witnesses. Its maximum-local interval is 10,342 cycles versus 10,542 sampled and 9,684 original-source control. Omitting full prefix stores reduces this local interval by about 1.9%; residual instrumentation overhead is about 6.8% relative to the native control. No claim that all diagnostic cost has been removed. See `evidence/twohop64-instrumentation-comparison.json`.

The generic frozen execution driver now accepts an explicit positive wall-time budget and preserves its own source alongside the bundle. This accommodates larger simulator jobs without changing code, precision or inputs; wall-clock runtime is not a CSL performance metric.
