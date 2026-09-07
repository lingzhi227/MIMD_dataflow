# Blocked distributed LU

`mesh_lu.v1` expresses the Matrix algorithms project's many-elements-per-PE
no-pivot LU in the HLS frontend. It preserves the source's complete preceding
block-pivot consumption, local diagonal elimination, row-signal and division
data tasks, and vector rank-1 updates. It is distinct from the earlier 4×4
equation-only profile.

```cpp
#include "spatial.hpp"
void design() {
  auto a = spatial::input<128,128>("a");
  #pragma csl dataflow rows=8 cols=8 pivot=none update=blocked fp=relaxed compute=vector
  auto result = spatial::lu_no_pivot(a);
  spatial::output("result",result);
}
```

The first admissible profile requires finite, strictly row diagonally dominant
input. This is narrower than all matrices admitting no-pivot LU. Native C++ and
the f32 interpreter reject unsupported inputs before SDK execution; they also
check pivots before dividing. No diagonal shift, equilibration or pivoting is
silently introduced. Tests include rejection of a nonsingular matrix needing
a pivot and acceptance of nonsymmetric dominant matrices.

The output packs the strict lower multipliers of L and the upper triangle of
U into one matrix. L's diagonal is implicitly one; it is a representation
convention, not an independently observed device value. Validation compares
the whole packed result against a separate left-looking float64 Doolittle
implementation using `math.fsum`, with fixed `rtol=3e-5, atol=3e-6`, and checks
the full infinity-norm relative reconstruction residual `L*U-A <= 3e-5`.

## Original algorithm and CSL migration

The pinned original is commit `016156e79b63fe45e118580da8db694285b6c6d9`,
`lu_factorization/many_elements_per_pe/{layout,pe_program}.csl`. Its legacy
multi-input routes require a WSE-3 adaptation. The prior independently executed
SDK 2.10.1 migration consumes all preceding pivot blocks through WEST/NORTH
routes, then switches a completed-prefix PE to RAMP injection. It retains the
original arithmetic and ordering. Reference copies and provenance are under
`experiments/reference/matrix-lu-sdk2101`; their hashes are recorded separately
from the original upstream source. That prior evidence was one-shot only.

The HLS runtime adds a global `prepare` barrier for repeatability. Preparation
restores memory DSD offsets/lengths, fabric DSD extents, counters, data-task
blocking state and the original receive/forward routes. The factor entrypoint
then starts its own source algorithm. Colors 0–3 carry row signals, horizontal
elimination, division pivots and vertical elimination. Data tasks use input
queues 4/5, panel DSDs use input queues 2/3, and output queues 4–7 are separate
from SDK memcpy's queues. Local task 10 runs the diagonal work.

Actual SDK compilation exposed an API distinction: dynamic
`tile_config.color_config.reset_routes` requires an explicit
`[2]direction{RAMP,EAST}` array for a multi-direction output; the anonymous
tuple accepted by layout configuration is not iterable by this library API.
The failed `run-20260906T102736736733Z` is preserved.

## Intermediate observations and execution

Two tile corners are recorded after every active global pivot. Independent
global float64 rank-1 updates reproduce these sampled states. Each PE retires
after `(min(column,row)+1)*tile_width` pivots; counters and timing are exported.
These are sampled witnesses, not complete panel/tile traces. The host only
packs/unpacks data and launches phases; it performs no elimination.

`run-20260906T102825505499Z` passed 32×32 on 4×4 PEs for four calls in one
runtime: dense dominant, diagonal-after-dense, differently scaled dominant,
and nonsymmetric tridiagonal. Independent coordinator reconstruction is in
`coordination/lu32-review.json`, maximum relative infinity residual
1.755233e-7. The 128×128 / 8×8 PE case `run-20260906T103038422731Z` also passed four
calls. Independent reconstruction is in `coordination/lu128-review.json`,
maximum relative infinity residual 3.060258e-7. Separate exact-zero checks of
unexecuted history are preserved in `evidence/lu128-checkpoint-unused-exact.json`.

`factor_sdk.py` shares row-major host packing and the prepare/factor lifecycle
with Cholesky. The Cholesky32 SDK regression `run-20260906T102901336270Z`
passed after this extraction. `toolchain/debug.py BUNDLE --node p2_3 --epoch 0
--step 12` inspects the expected and observed local pivot witness.

## Performance scope and remaining work

`experiments/factor_sdk_baseline.py BUNDLE` compares the first input with the
explicitly identified SDK 2.10.1 migrated original, adding timestamps only to
that baseline. It does not call the migrated baseline an HLS port or an
unmodified upstream execution. It retains source hashes, complete outputs,
fixed numerical checks and all per-PE intervals. Timing excludes preparation
and host I/O; it is not synchronized end-to-end latency or hardware speedup.
Completed comparisons are `evidence/native-lu-20260906T103609070177Z` (32)
and `evidence/native-lu-20260906T103627610307Z` (128). Maximum local factor
intervals are HLS/baseline 37,665/36,948 (1.01941) and 337,088/333,229
(1.01158), respectively. The complete per-PE arrays are retained. These are
bounded simulator observations with HLS diagnostics included; they do not
bound preparation cost or full-call latency. Both baseline outputs pass the
unchanged factor/reconstruction checks.

This profile supports square 2–8 PE meshes and divisible square tiles of width
at least two. General pivoting, arbitrary admissible matrices, uneven blocks,
factor reuse/triangular solve and graph composition remain outside its scope.
